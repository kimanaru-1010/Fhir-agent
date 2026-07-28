"use client";

export interface ParsedSseEvent {
  event: string;
  data: unknown;
}

export async function* parseSseStream(stream: ReadableStream<Uint8Array>): AsyncGenerator<ParsedSseEvent> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      yield* drainEventsFromBuffer(() => buffer, (next) => {
        buffer = next;
      });
    }

    buffer += decoder.decode();
    if (buffer.trim()) {
      buffer += "\n\n";
      yield* drainEventsFromBuffer(() => buffer, (next) => {
        buffer = next;
      });
    }
  } finally {
    reader.releaseLock();
  }
}

function* drainEventsFromBuffer(
  getBuffer: () => string,
  setBuffer: (value: string) => void,
): Generator<ParsedSseEvent> {
  let buffer = getBuffer();
  let separator = buffer.indexOf("\n\n");
  while (separator !== -1) {
    const block = buffer.slice(0, separator);
    buffer = buffer.slice(separator + 2);
    const parsed = parseEventBlock(block);
    if (parsed) yield parsed;
    separator = buffer.indexOf("\n\n");
  }
  setBuffer(buffer);
}

function parseEventBlock(block: string): ParsedSseEvent | null {
  let event = "message";
  const dataLines: string[] = [];

  for (const rawLine of block.split("\n")) {
    const line = rawLine.endsWith("\r") ? rawLine.slice(0, -1) : rawLine;
    if (!line || line.startsWith(":")) continue;
    if (line.startsWith("event:")) {
      event = line.slice("event:".length).trim();
    } else if (line.startsWith("data:")) {
      dataLines.push(line.slice("data:".length).trimStart());
    }
  }

  if (dataLines.length === 0) return null;
  try {
    return {
      event,
      data: JSON.parse(dataLines.join("\n")) as unknown,
    };
  } catch {
    return null;
  }
}
