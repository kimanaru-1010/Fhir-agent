"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import {
  Box, Flex, Heading, Text, Textarea, IconButton, VStack, HStack,
  Badge, Button, Spinner, Skeleton, Collapsible, Timeline, Circle,
  Input,
} from "@chakra-ui/react";
import {
  Send, RotateCcw, ChevronDown, Wrench, Check, Bot, User, Sparkles,
  Plus, Trash2, LogOut, ImagePlus, X,
} from "lucide-react";
import ReactMarkdown from "react-markdown";
import type { Components } from "react-markdown";
import remarkGfm from "remark-gfm";
import { API_BASE, DOMAIN } from "@/lib/config";
import type { GraphData } from "@/lib/config";
import {
  ApiError,
  clearAuth,
  deleteConversation,
  getAccessToken,
  getStoredUser,
  getSkinDiagnosticStatus,
  createImageUploadConversation,
  listConversations,
  listMessages,
  login,
  openConversationStream,
  openMessageStream,
  register,
  sendImageUploadMessage,
  submitSkinDiagnosticAnswers,
} from "@/lib/api";

import type {
  ChatAttachment,
  ChatImageAttachment,
  ChatMessage,
  Conversation,
  SkinDiagnosticResult,
  SkinDiagnosticResultAttachment,
  SkinDiagnosticStatus,
  SkinPendingQuestion,
  UserProfile,
} from "@/lib/api";
import { parseSseStream } from "@/lib/sse";
import type { ParsedSseEvent } from "@/lib/sse";

const CHAT_STREAM_TIMEOUT_MS = 900_000;

const markdownComponents: Components = {
  img({ src, alt, ...props }) {
    if (typeof src !== "string" || !src.trim()) return null;
    return <img src={src} alt={alt || ""} {...props} />;
  },
};

interface ToolCall {
  name: string;
  inputs: Record<string, unknown>;
  output_preview: string;
  status: "running" | "complete" | "failed";
  graph_data?: GraphData;
  raw_output?: unknown;
}

interface ExtractedEntity {
  name: string;
  type: string;
  subtype?: string;
}

interface DetectedPreference {
  category: string;
  preference: string;
  confidence?: number;
}

interface Message extends ChatMessage {
  role: "user" | "assistant" | "system";
  toolCalls?: ToolCall[];
  retryInput?: string;
  pending?: boolean;
  failed?: boolean;
  entities?: ExtractedEntity[];
  preferences?: DetectedPreference[];
  attachments?: ChatAttachment[];
  diagnosticRunId?: string | null;
}

interface ChatInterfaceProps {
  onGraphUpdate?: (data: GraphData) => void;
  externalInput?: string | null;
  onExternalInputConsumed?: () => void;
}

const THINKING_PATTERNS = [
  /^let me /i, /^i'll /i, /^i will /i, /^first,? i /i,
  /^now let me /i, /^let me also /i, /^let me try /i,
  /^i need to /i, /^i should /i, /^let me check /i,
  /^let me look /i, /^let me search /i, /^let me query /i,
  /^let me find /i, /^now i'll /i, /^now i need /i,
];

const CONTINUATION_PATTERNS = [
  /^(and |also |then |additionally |next |finally )/i,
  /^(this will |this should |this means |that way )/i,
  /^(so |because |since |in order to )/i,
  /^(after that |once |before )/i,
];

const MARKDOWN_LINE = /^(#{1,6} |[-*] |\d+\. |\|)/;

function splitThinkingAndResponse(text: string): { thinking: string; response: string } {
  if (/\berror\b/i.test(text) || /\bfailed\b/i.test(text) || /\bsyntax error\b/i.test(text)) {
    return { thinking: "", response: text };
  }

  const lines = text.split("\n");
  const thinkingLines: string[] = [];
  const responseLines: string[] = [];
  let foundResponse = false;
  let inThinkingBlock = false;

  for (const line of lines) {
    const trimmed = line.trim();
    if (!foundResponse && trimmed && THINKING_PATTERNS.some((p) => p.test(trimmed))) {
      thinkingLines.push(line);
      inThinkingBlock = true;
    } else if (
      inThinkingBlock &&
      !foundResponse &&
      trimmed &&
      !MARKDOWN_LINE.test(trimmed) &&
      (CONTINUATION_PATTERNS.some((p) => p.test(trimmed)) || trimmed.length < 80)
    ) {
      thinkingLines.push(line);
    } else {
      if (trimmed) {
        foundResponse = true;
        inThinkingBlock = false;
      }
      responseLines.push(line);
    }
  }

  const response = responseLines.join("\n").trim();
  const thinking = thinkingLines.join("\n").trim();
  if (!response && thinking) return { thinking: "", response: text };
  return { thinking, response };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function asString(value: unknown): string {
  return typeof value === "string" ? value : "";
}

function isConversation(value: unknown): value is Conversation {
  return isRecord(value) && typeof value.id === "string" && typeof value.title === "string";
}

function isChatMessage(value: unknown): value is ChatMessage {
  return (
    isRecord(value) &&
    typeof value.id === "string" &&
    typeof value.conversation_id === "string" &&
    typeof value.role === "string" &&
    typeof value.content === "string" &&
    typeof value.created_at === "string"
  );
}

function isChatImageAttachment(value: unknown): value is ChatImageAttachment {
  return (
    isRecord(value) &&
    value.type === "image" &&
    typeof value.patient_id === "string" &&
    typeof value.diagnostic_report_id === "string" &&
    typeof value.media_id === "string" &&
    typeof value.binary_id === "string" &&
    typeof value.url === "string"
  );
}

function isSkinDiagnosticResultAttachment(value: unknown): value is SkinDiagnosticResultAttachment {
  return (
    isRecord(value) &&
    value.type === "skin_diagnostic_result" &&
    typeof value.run_id === "string" &&
    isRecord(value.result)
  );
}

function isGraphData(value: unknown): value is GraphData {
  return isRecord(value) && Array.isArray(value.results);
}

function mapBackendMessage(message: ChatMessage): Message {
  return {
    id: message.id,
    conversation_id: message.conversation_id,
    role: message.role,
    content: message.content,
    message_type: message.message_type ?? "text",
    created_at: message.created_at,
    attachments: message.attachments,
    diagnosticRunId: message.diagnostic_run_id ?? null,
  };
}

function upsertConversation(items: Conversation[], conversation: Conversation): Conversation[] {
  return [conversation, ...items.filter((item) => item.id !== conversation.id)];
}

function replaceMessage(messages: Message[], localId: string, message: ChatMessage): Message[] {
  return messages.map((item) => (
    item.id === localId ? { ...mapBackendMessage(message), toolCalls: item.toolCalls } : item
  ));
}

function appendMessageOnce(messages: Message[], message: Message): Message[] {
  if (messages.some((item) => item.id === message.id)) return messages;
  return [...messages, message];
}

function stepLabel(step: string): string {
  const labels: Record<string, string> = {
    visual_extract: "Trích xuất đặc điểm hình ảnh",
    knowledge_base: "Tra cứu tri thức bệnh lý",
    clinical_planner_round1: "Lập câu hỏi lâm sàng 1",
    user_interview_round1: "Hỏi bệnh 1",
    clinical_planner_round2: "Lập câu hỏi lâm sàng 2",
    user_interview_round2: "Hỏi bệnh 2",
    diagnostic_reasoning: "Suy luận chẩn đoán",
  };
  return labels[step] || step || "Đang chờ";
}

function hasDiagnosticResult(status: SkinDiagnosticStatus | null): status is SkinDiagnosticStatus & { result: SkinDiagnosticResult } {
  return Boolean(
    status &&
    status.status === "completed" &&
    "ranked_diagnoses" in status.result,
  );
}

function confidenceLabel(confidence: string): string {
  const labels: Record<string, string> = {
    High: "Kha nang cao",
    Medium: "Kha nang trung binh",
    Low: "Can xem xet",
  };

  return labels[confidence] || confidence || "Chua xac dinh";
}

function confidencePalette(
  confidence: string,
): "green" | "orange" | "blue" | "gray" {
  const palettes: Record<
    string,
    "green" | "orange" | "blue" | "gray"
  > = {
    High: "green",
    Medium: "orange",
    Low: "blue",
  };

  return palettes[confidence] || "gray";
}

function diagnosisBorderColor(confidence: string): string {
  const colors: Record<string, string> = {
    High: "green.400",
    Medium: "orange.400",
    Low: "blue.400",
  };

  return colors[confidence] || "gray.300";
}

function extractSkinDiagnosticRunIds(toolCalls?: ToolCall[]): string[] {
  if (!toolCalls?.length) return [];
  const runIds: string[] = [];
  for (const toolCall of toolCalls) {
    if (toolCall.name !== "start_skin_diagnostic" || !toolCall.output_preview) continue;
    try {
      const payload = JSON.parse(toolCall.output_preview) as {
        status?: string;
        data?: { run_id?: string };
      };
      const runId = payload.status === "ok" ? payload.data?.run_id : "";
      if (runId && !runIds.includes(runId)) runIds.push(runId);
    } catch {
      continue;
    }
  }
  return runIds;
}

export function ChatInterface({ onGraphUpdate, externalInput, onExternalInputConsumed }: ChatInterfaceProps) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeConversationId, setActiveConversationId] = useState<string | null>(null);
  const [user, setUser] = useState<UserProfile | null>(null);
  const [authReady, setAuthReady] = useState(false);
  const [authMode, setAuthMode] = useState<"login" | "register">("login");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [authLoading, setAuthLoading] = useState(false);
  const [loadingConversations, setLoadingConversations] = useState(false);
  const [loadingMessages, setLoadingMessages] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [chatImage, setChatImage] = useState<File | null>(null);
  const [chatImagePatientId, setChatImagePatientId] = useState("");
  const [uploadingImage, setUploadingImage] = useState(false);
  const [streamingContent, setStreamingContent] = useState("");
  const [streamingToolCalls, setStreamingToolCalls] = useState<ToolCall[]>([]);
  const [streamingEntities, setStreamingEntities] = useState<ExtractedEntity[]>([]);
  const [streamingPreferences, setStreamingPreferences] = useState<DetectedPreference[]>([]);
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textBufferRef = useRef("");
  const flushTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);
  const imageInputRef = useRef<HTMLInputElement>(null);
  const streamingEntitiesRef = useRef<ExtractedEntity[]>([]);
  const streamingPreferencesRef = useRef<DetectedPreference[]>([]);

  const handleAuthFailure = useCallback(() => {
    clearAuth();
    setUser(null);
    setMessages([]);
    setConversations([]);
    setActiveConversationId(null);
    setError("Please sign in again.");
  }, []);

  const loadConversationList = useCallback(async () => {
    if (!getAccessToken()) return;
    setLoadingConversations(true);
    setError(null);
    try {
      const data = await listConversations();
      setConversations(data.items);
    } catch (err) {
      if (err instanceof ApiError && err.status === 401) {
        handleAuthFailure();
      } else {
        setError(err instanceof Error ? err.message : "Unable to load conversations");
      }
    } finally {
      setLoadingConversations(false);
    }
  }, [handleAuthFailure]);

  const loadConversationMessages = useCallback(async (conversationId: string) => {
    setActiveConversationId(conversationId);
    setMessages([]);
    setStreamingContent("");
    setStreamingToolCalls([]);
    setLoadingMessages(true);
    setError(null);
    try {
      const data = await listMessages(conversationId);
      setMessages(data.items.map(mapBackendMessage));
    } catch (err) {
      if (err instanceof ApiError && err.status === 401) {
        handleAuthFailure();
      } else if (err instanceof ApiError && err.status === 404) {
        setActiveConversationId(null);
        setMessages([]);
        await loadConversationList();
        setError("Conversation not found.");
      } else {
        setError(err instanceof Error ? err.message : "Unable to load messages");
      }
    } finally {
      setLoadingMessages(false);
    }
  }, [handleAuthFailure, loadConversationList]);

  useEffect(() => {
    const storedUser = getStoredUser();
    if (getAccessToken() && storedUser) {
      setUser(storedUser);
      void loadConversationList();
    }
    setAuthReady(true);
  }, [loadConversationList]);

  useEffect(() => {
    if (externalInput && !loading && user) {
      sendMessage(externalInput);
      onExternalInputConsumed?.();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [externalInput, loading, user]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, streamingContent, streamingToolCalls]);

  useEffect(() => {
    if (!loading) { setElapsedSeconds(0); return; }
    const interval = setInterval(() => setElapsedSeconds((s) => s + 1), 1000);
    return () => clearInterval(interval);
  }, [loading]);

  const flushTextBuffer = useCallback(() => {
    setStreamingContent(textBufferRef.current);
    flushTimerRef.current = null;
  }, []);

  const appendStreamingText = useCallback((text: string) => {
    textBufferRef.current += text;
    if (!flushTimerRef.current) {
      flushTimerRef.current = setTimeout(flushTextBuffer, 50);
    }
  }, [flushTextBuffer]);

  function cancelRequest() {
    abortControllerRef.current?.abort();
    abortControllerRef.current = null;
  }

  function resetStreamingState() {
    setStreamingContent("");
    setStreamingToolCalls([]);
    setStreamingEntities([]);
    setStreamingPreferences([]);
    streamingEntitiesRef.current = [];
    streamingPreferencesRef.current = [];
    textBufferRef.current = "";
  }

  function clearChatImage() {
    setChatImage(null);
    if (imageInputRef.current) imageInputRef.current.value = "";
  }

  function startNewConversation() {
    cancelRequest();
    setActiveConversationId(null);
    setMessages([]);
    resetStreamingState();
    setLoading(false);
    setError(null);
  }

  async function handleAuthSubmit() {
    const trimmedUsername = username.trim();
    if (!trimmedUsername || !password || authLoading) return;
    setAuthLoading(true);
    setError(null);
    try {
      if (authMode === "register") {
        await register(trimmedUsername, password);
      }
      const tokenResponse = await login(trimmedUsername, password);
      setUser(tokenResponse.user);
      setUsername("");
      setPassword("");
      await loadConversationList();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Authentication failed");
    } finally {
      setAuthLoading(false);
    }
  }

  function handleLogout() {
    cancelRequest();
    clearAuth();
    setUser(null);
    setConversations([]);
    setActiveConversationId(null);
    setMessages([]);
    resetStreamingState();
    setError(null);
  }

  function applyToolStart(data: Record<string, unknown>, toolCalls: ToolCall[]): ToolCall[] {
    return [
      ...toolCalls,
      {
        name: asString(data.name) || "tool",
        inputs: isRecord(data.inputs) ? data.inputs : {},
        output_preview: "",
        status: "running",
      },
    ];
  }

  function applyToolEnd(data: Record<string, unknown>, toolCalls: ToolCall[]): ToolCall[] {
    const endName = asString(data.name);
    let matched = false;
    return toolCalls.map((tc) => {
      if (tc.name === endName && tc.status === "running" && !matched) {
        matched = true;
        const graphData = isGraphData(data.graph_data) ? data.graph_data : undefined;
        return {
          ...tc,
          output_preview: asString(data.output_preview),
          status: "complete" as const,
          graph_data: graphData,
          raw_output: data,
        };
      }
      return tc;
    });
  }

  async function sendMessage(text?: string) {
    const messageText = text || input.trim();
    if ((!messageText && !chatImage) || loading || uploadingImage || !user) return;
    if (chatImage) {
      await uploadImageFromChat(messageText);
      return;
    }

    const targetConversationId = activeConversationId;
    const localUserId = `local-user-${crypto.randomUUID()}`;
    const userMessage: Message = {
      id: localUserId,
      conversation_id: targetConversationId ?? "pending",
      role: "user",
      content: messageText,
      created_at: new Date().toISOString(),
      pending: true,
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setLoading(true);
    setError(null);
    resetStreamingState();

    const controller = new AbortController();
    abortControllerRef.current = controller;
    let timeout = setTimeout(
      () => controller.abort(),
      CHAT_STREAM_TIMEOUT_MS,
    );
    const resetTimeout = () => {
      clearTimeout(timeout);
      timeout = setTimeout(
        () => controller.abort(),
        CHAT_STREAM_TIMEOUT_MS,
      );
    };

    let fullText = "";
    let toolCalls: ToolCall[] = [];
    let confirmedUserId: string | null = null;

    try {
      const response = targetConversationId
        ? await openMessageStream(targetConversationId, messageText, controller.signal)
        : await openConversationStream(messageText, controller.signal);

      if (!response.ok) {
        const errorData = await response.json().catch(() => null);
        const detail = isRecord(errorData) ? asString(errorData.detail) : "";
        throw new Error(detail || `Backend error (${response.status})`);
      }

      if (!response.body) {
        throw new Error("No response body for streaming");
      }

      for await (const parsedEvent of parseSseStream(response.body)) {
        resetTimeout();
        await handleChatEvent(parsedEvent);
      }
    } catch (err: unknown) {
      if (err instanceof ApiError && err.status === 401) {
        handleAuthFailure();
      }
      if (flushTimerRef.current) {
        clearTimeout(flushTimerRef.current);
        flushTimerRef.current = null;
      }
      const errorMsg = err instanceof DOMException && err.name === "AbortError"
        ? "Request timed out or was cancelled. Please try again."
        : err instanceof Error && err.message
          ? err.message
          : "Cannot reach the backend. Is it running?";
      setMessages((prev) => {
        const marked = confirmedUserId
          ? prev
          : prev.map((item) => item.id === localUserId ? { ...item, failed: true, pending: false } : item);
        return [
          ...marked,
          {
            id: `local-error-${crypto.randomUUID()}`,
            conversation_id: activeConversationId ?? confirmedUserId ?? "pending",
            role: "assistant",
            content: `**Error:** ${errorMsg}`,
            created_at: new Date().toISOString(),
            retryInput: messageText,
          },
        ];
      });
      toolCalls = toolCalls.map((tc) => tc.status === "running" ? { ...tc, status: "failed" } : tc);
      setStreamingToolCalls([]);
      setStreamingContent("");
      textBufferRef.current = "";
    } finally {
      clearTimeout(timeout);
      abortControllerRef.current = null;
      setLoading(false);
    }

    async function handleChatEvent({ event, data }: ParsedSseEvent) {
      if (!isRecord(data)) return;

      switch (event) {
        case "conversation_started": {
          const conversation = data.conversation;
          const userMessageData = data.user_message;
          if (isConversation(conversation)) {
            setActiveConversationId(conversation.id);
            setConversations((prev) => upsertConversation(prev, conversation));
          }
          if (isChatMessage(userMessageData)) {
            confirmedUserId = userMessageData.conversation_id;
            setMessages((prev) => replaceMessage(prev, localUserId, userMessageData));
          }
          break;
        }

        case "message_started": {
          const userMessageData = data.user_message;
          if (isChatMessage(userMessageData)) {
            confirmedUserId = userMessageData.conversation_id;
            setMessages((prev) => replaceMessage(prev, localUserId, userMessageData));
          }
          break;
        }

        case "tool_start":
          toolCalls = applyToolStart(data, toolCalls);
          setStreamingToolCalls([...toolCalls]);
          break;

        case "tool_end": {
          toolCalls = applyToolEnd(data, toolCalls);
          setStreamingToolCalls([...toolCalls]);
          const graphData = isGraphData(data.graph_data) ? data.graph_data : undefined;
          if (graphData?.results?.length && onGraphUpdate) {
            onGraphUpdate(graphData);
          }
          break;
        }

        case "text_delta": {
          const delta = asString(data.text) || asString(data.delta);
          fullText += delta;
          appendStreamingText(delta);
          break;
        }

        case "entities_extracted":
          if (Array.isArray(data.entities)) {
            streamingEntitiesRef.current = [
              ...streamingEntitiesRef.current,
              ...(data.entities as ExtractedEntity[]),
            ];
            setStreamingEntities([...streamingEntitiesRef.current]);
          }
          break;

        case "preferences_detected":
          if (Array.isArray(data.preferences)) {
            streamingPreferencesRef.current = [
              ...streamingPreferencesRef.current,
              ...(data.preferences as DetectedPreference[]),
            ];
            setStreamingPreferences([...streamingPreferencesRef.current]);
          }
          break;

        case "done": {
          if (flushTimerRef.current) {
            clearTimeout(flushTimerRef.current);
            flushTimerRef.current = null;
          }
          const conversation = data.conversation;
          const userMessageData = data.user_message;
          const assistantMessage = data.assistant_message;
          const responseText = asString(data.response) || fullText;
          const attachments = Array.isArray(data.attachments)
            ? data.attachments.filter(isChatImageAttachment)
            : [];

          if (isConversation(conversation)) {
            setActiveConversationId(conversation.id);
            setConversations((prev) => upsertConversation(prev, conversation));
          } else {
            void loadConversationList();
          }

          if (isChatMessage(userMessageData)) {
            setMessages((prev) => replaceMessage(prev, localUserId, userMessageData));
          }

          if (isChatMessage(assistantMessage)) {
            const finalEntities = streamingEntitiesRef.current;
            const finalPreferences = streamingPreferencesRef.current;
            setMessages((prev) => appendMessageOnce(prev, {
              ...mapBackendMessage(assistantMessage),
              content: responseText || assistantMessage.content,
              toolCalls: toolCalls.length > 0 ? [...toolCalls] : undefined,
              entities: finalEntities.length > 0 ? [...finalEntities] : undefined,
              preferences: finalPreferences.length > 0 ? [...finalPreferences] : undefined,
              attachments: attachments.length > 0 ? attachments : undefined,
            }));
          }
          resetStreamingState();
          break;
        }

        case "error":
          throw new Error(asString(data.detail) || "Streaming error");
      }
    }
  }

  async function uploadImageFromChat(messageText: string) {
    if (!chatImage || !user) return;

    const patientId = chatImagePatientId.trim();
    if (!patientId) {
      setError("Enter Patient ID before uploading the image.");
      return;
    }

    const localAssistantId = `local-assistant-${crypto.randomUUID()}`;
    const userContent = messageText;

    setUploadingImage(true);
    setLoading(true);
    setError(null);

    try {
      const result = activeConversationId
        ? await sendImageUploadMessage(activeConversationId, chatImage, patientId, userContent)
        : await createImageUploadConversation(chatImage, patientId, userContent);

      setActiveConversationId(result.conversation_id);
      void loadConversationList();
      setMessages((previous) => [
        ...previous,
        mapBackendMessage(result.user_message),
        {
          ...mapBackendMessage(result.assistant_message),
          diagnosticRunId: result.diagnostic_run_id ?? null,
        },
      ]);

      setInput("");
      setChatImagePatientId("");
      clearChatImage();
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          id: localAssistantId,
          conversation_id: activeConversationId ?? "local-upload",
          role: "assistant",
          content: `**Error:** ${err instanceof Error ? err.message : "Unable to upload image"}`,
          created_at: new Date().toISOString(),
          failed: true,
        },
      ]);
      setError(err instanceof Error ? err.message : "Unable to upload image");
    } finally {
      setUploadingImage(false);
      setLoading(false);
    }
  }

  async function handleDeleteConversation(conversationId: string) {
    if (loading) return;
    setError(null);
    try {
      await deleteConversation(conversationId);
      const remaining = conversations.filter((item) => item.id !== conversationId);
      setConversations(remaining);
      if (activeConversationId === conversationId) {
        const next = remaining[0];
        if (next) {
          await loadConversationMessages(next.id);
        } else {
          startNewConversation();
        }
      }
    } catch (err) {
      if (err instanceof ApiError && err.status === 401) {
        handleAuthFailure();
      } else {
        setError(err instanceof Error ? err.message : "Unable to delete conversation");
      }
    }
  }

  function handleKeyDown(e: React.KeyboardEvent) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  }

  const persistedSkinDiagnosticRunIds = new Set(
    messages.flatMap((message) =>
      (message.attachments || [])
        .filter(isSkinDiagnosticResultAttachment)
        .map((attachment) => attachment.run_id),
    ),
  );

  if (!authReady) {
    return (
      <Flex align="center" justify="center" h="100%">
        <Spinner size="sm" />
      </Flex>
    );
  }

  if (!user) {
    return (
      <Flex direction="column" h="100%">
        <HStack px={4} py={3} borderBottom="1px solid" borderColor="gray.200" justifyContent="space-between">
          <Heading size="sm">Chat</Heading>
        </HStack>
        <Flex flex={1} align="center" justify="center" px={4}>
          <VStack gap={3} align="stretch" w="100%" maxW="360px">
            <Heading size="sm">{authMode === "login" ? "Sign in" : "Create account"}</Heading>
            <Input
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              placeholder="Username"
              autoComplete="username"
            />
            <Input
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="Password"
              type="password"
              autoComplete={authMode === "login" ? "current-password" : "new-password"}
            />
            {error && <Text color="red.500" fontSize="sm">{error}</Text>}
            <Button
              colorPalette="blue"
              onClick={handleAuthSubmit}
              loading={authLoading}
              disabled={
                !username.trim() ||
                !password
              }
            >
              {authMode === "login" ? "Sign in" : "Create and sign in"}
            </Button>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => {
                setAuthMode(authMode === "login" ? "register" : "login");
                setError(null);
              }}
            >
              {authMode === "login" ? "Create account" : "Use existing account"}
            </Button>
          </VStack>
        </Flex>
      </Flex>
    );
  }

  return (
    <Flex h="100%">
      <Box w="150px" borderRight="1px solid" borderColor="gray.200" overflow="auto" flexShrink={0}>
        <HStack px={2} py={2} justify="space-between">
          <Text fontSize="xs" fontWeight="medium" color="gray.600">Chats</Text>
          <IconButton aria-label="New chat" size="xs" variant="ghost" onClick={startNewConversation}>
            <Plus size={14} />
          </IconButton>
        </HStack>
        {loadingConversations ? (
          <VStack align="stretch" px={2} gap={2}>
            <Skeleton height="6" />
            <Skeleton height="6" />
          </VStack>
        ) : (
          <VStack align="stretch" px={1} gap={1}>
            {conversations.map((conversation) => (
              <HStack
                key={conversation.id}
                gap={1}
                px={2}
                py={1.5}
                borderRadius="md"
                bg={activeConversationId === conversation.id ? "blue.50" : "transparent"}
                _hover={{ bg: "gray.50" }}
              >
                <Button
                  variant="ghost"
                  size="xs"
                  justifyContent="flex-start"
                  flex={1}
                  minW={0}
                  onClick={() => loadConversationMessages(conversation.id)}
                >
                  <Text truncate fontSize="xs">{conversation.title}</Text>
                </Button>
                <IconButton
                  aria-label="Delete conversation"
                  size="2xs"
                  variant="ghost"
                  color="gray.500"
                  onClick={() => handleDeleteConversation(conversation.id)}
                >
                  <Trash2 size={12} />
                </IconButton>
              </HStack>
            ))}
          </VStack>
        )}
      </Box>

      <Flex direction="column" h="100%" flex={1} minW={0}>
        <HStack px={4} py={3} borderBottom="1px solid" borderColor="gray.200" justifyContent="space-between">
          <Heading size="sm">Chat</Heading>
          <HStack gap={1}>
            {(messages.length > 0 || activeConversationId) && (
              <Button size="xs" variant="ghost" onClick={startNewConversation}>
                <RotateCcw size={14} />
                New
              </Button>
            )}
            <IconButton aria-label="Sign out" size="xs" variant="ghost" onClick={handleLogout}>
              <LogOut size={14} />
            </IconButton>
          </HStack>
        </HStack>

        {error && (
          <Box px={4} py={2} bg="red.50" color="red.700" fontSize="sm">
            {error}
          </Box>
        )}

        {loadingMessages && (
          <HStack px={4} py={2} gap={2} color="gray.500">
            <Spinner size="xs" />
            <Text fontSize="sm">Loading messages...</Text>
          </HStack>
        )}

        {messages.length === 0 && !loading && !loadingMessages && (
          <Flex direction="column" flex={1} justify="center" px={4} py={4} minH={0}>
            <VStack gap={2}>
              <Text fontSize="lg" fontWeight="medium" color="gray.700">
                How can I help you?
              </Text>
              <Button
                size="xs"
                variant="outline"
                onClick={() => sendMessage("Show me all patients with a chronic diagnosis")}
              >
                <Sparkles size={14} />
                Try a sample prompt
              </Button>
            </VStack>
          </Flex>
        )}

        <VStack flex={1} overflow="auto" px={4} py={2} gap={3} align="stretch"
          display={messages.length === 0 && !loading ? "none" : "flex"}
        >
          {messages.map((msg) => (
            <Box key={msg.id}>
              {msg.toolCalls && msg.toolCalls.length > 0 && (
                <ToolCallTimeline toolCalls={msg.toolCalls} />
              )}
              <Flex gap={2} alignItems="flex-start">
                <Circle
                  size="7"
                  bg={msg.role === "user" ? "blue.500" : "gray.600"}
                  color="white"
                  flexShrink={0}
                  mt={0.5}
                >
                  {msg.role === "user" ? <User size={14} /> : <Bot size={14} />}
                </Circle>
                <Box
                  bg={msg.role === "user" ? "blue.50" : "gray.50"}
                  px={3}
                  py={2}
                  borderRadius="lg"
                  flex={1}
                  maxW="95%"
                >
                  {msg.role === "assistant" ? (
                    <Box fontSize="sm" className="markdown-content">
                      {(() => {
                        const resultAttachment = (msg.attachments || []).find(isSkinDiagnosticResultAttachment);
                        if (msg.message_type === "skin_diagnostic_result" && resultAttachment) {
                          return (
                            <SkinDiagnosticResultMessageCard
                              runId={resultAttachment.run_id}
                              result={resultAttachment.result}
                            />
                          );
                        }
                        const { thinking, response } = splitThinkingAndResponse(msg.content);
                        return (
                          <>
                            {thinking && (
                              <Collapsible.Root>
                                <Collapsible.Trigger asChild>
                                  <Button variant="ghost" size="xs" mb={1} color="gray.500">
                                    <ChevronDown size={12} />
                                    Show reasoning
                                  </Button>
                                </Collapsible.Trigger>
                                <Collapsible.Content>
                                  <Box px={2} py={1} mb={2} bg="gray.100" borderRadius="sm" fontSize="xs" color="gray.600">
                                    <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents}>{thinking}</ReactMarkdown>
                                  </Box>
                                </Collapsible.Content>
                              </Collapsible.Root>
                            )}
                            <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents}>{response || msg.content}</ReactMarkdown>
                          </>
                        );
                      })()}
                      {msg.entities && msg.entities.length > 0 && (
                        <HStack gap={1} mt={2} flexWrap="wrap">
                          {msg.entities.map((e, i) => (
                            <Badge key={`${e.type}-${e.name}-${i}`} size="xs" colorPalette="teal" variant="subtle">
                              {e.type}{e.subtype ? `/${e.subtype}` : ""}: {e.name}
                            </Badge>
                          ))}
                        </HStack>
                      )}
                      {msg.preferences && msg.preferences.length > 0 && (
                        <HStack gap={1} mt={1} flexWrap="wrap">
                          {msg.preferences.map((p, i) => (
                            <Badge key={`${p.category}-${p.preference}-${i}`} size="xs" colorPalette="orange" variant="subtle">
                              {p.category}: {p.preference}
                            </Badge>
                          ))}
                        </HStack>
                      )}
                      {msg.retryInput && (
                        <Button
                          size="xs"
                          variant="outline"
                          mt={2}
                          onClick={() => {
                            setMessages((prev) => prev.filter((m) => m.id !== msg.id));
                            sendMessage(msg.retryInput);
                          }}
                        >
                          <RotateCcw size={12} />
                          Retry
                        </Button>
                      )}
                    </Box>
                  ) : (
                    <Text fontSize="sm" whiteSpace="pre-wrap" color={msg.failed ? "red.600" : undefined}>
                      {msg.content}
                    </Text>
                  )}
                  {msg.attachments && msg.attachments.length > 0 && (
                    <VStack align="stretch" gap={3} mt={3}>
                      {msg.attachments.filter(isChatImageAttachment).map((attachment) => (
                        <SkinImageAttachmentView
                          key={`${attachment.diagnostic_report_id}-${attachment.binary_id}`}
                          attachment={attachment}
                          role={msg.role}
                          messageContent={msg.content}
                        />
                      ))}
                    </VStack>
                  )}
                  {[...new Set([
                    ...(msg.diagnosticRunId ? [msg.diagnosticRunId] : []),
                    ...extractSkinDiagnosticRunIds(msg.toolCalls),
                  ])]
                    .filter((runId) => !persistedSkinDiagnosticRunIds.has(runId))
                    .map((runId) => (
                    <ChatSkinDiagnosticRunCard
                      key={runId}
                      runId={runId}
                      onCompleted={() => {
                        if (activeConversationId) {
                          void loadConversationMessages(activeConversationId);
                        }
                      }}
                    />
                  ))}
                </Box>
              </Flex>
            </Box>
          ))}

          {loading && (
            <Box>
              {streamingToolCalls.length > 0 && (
                <ToolCallTimeline toolCalls={streamingToolCalls} />
              )}
              {streamingContent ? (
                <Flex gap={2} alignItems="flex-start">
                  <Circle size="7" bg="gray.600" color="white" flexShrink={0} mt={0.5}>
                    <Bot size={14} />
                  </Circle>
                  <Box bg="gray.50" px={3} py={2} borderRadius="lg" flex={1}>
                    <Box fontSize="sm" className="markdown-content">
                      <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents}>
                        {streamingContent}
                      </ReactMarkdown>
                    </Box>
                  </Box>
                </Flex>
              ) : (
                <Flex gap={2} alignItems="flex-start">
                  <Circle size="7" bg="gray.600" color="white" flexShrink={0} mt={0.5}>
                    <Bot size={14} />
                  </Circle>
                  <Box bg="gray.50" px={3} py={2} borderRadius="lg" flex={1}>
                    {streamingToolCalls.length === 0 ? (
                      <VStack align="stretch" gap={2}>
                        <HStack gap={2}>
                          <Spinner size="xs" />
                          <Text fontSize="sm" color="gray.500">Thinking...</Text>
                          {elapsedSeconds > 3 && (
                            <Text fontSize="xs" color="gray.400">{elapsedSeconds}s</Text>
                          )}
                        </HStack>
                        <Skeleton height="4" width="80%" />
                        <Skeleton height="4" width="60%" />
                      </VStack>
                    ) : (
                      <HStack gap={2}>
                        <Spinner size="xs" />
                        <Text fontSize="sm" color="gray.500">
                          Running tool {streamingToolCalls.filter(tc => tc.status === "complete").length + 1}
                          {" of "}
                          {streamingToolCalls.length}...
                        </Text>
                        {elapsedSeconds > 3 && (
                          <Text fontSize="xs" color="gray.400">{elapsedSeconds}s</Text>
                        )}
                      </HStack>
                    )}
                    <Button
                      size="xs"
                      variant="ghost"
                      mt={2}
                      onClick={cancelRequest}
                      color="gray.500"
                    >
                      Cancel
                    </Button>
                  </Box>
                </Flex>
              )}
            </Box>
          )}
          <div ref={messagesEndRef} />
        </VStack>

        <Box px={4} pt={3} pb="72px" borderTop="1px solid" borderColor="gray.200">
          <Box mb={2}>
            <HStack gap={2} align="center" flexWrap="wrap">
              <input
                ref={imageInputRef}
                type="file"
                accept="image/png,image/jpeg,image/webp"
                style={{ display: "none" }}
                onChange={(event) => setChatImage(event.target.files?.[0] ?? null)}
              />
              <Button
                size="sm"
                variant="outline"
                onClick={() => imageInputRef.current?.click()}
                disabled={loading || uploadingImage}
              >
                <ImagePlus size={16} />
                Upload image
              </Button>
              {chatImage && (
                <>
                  <Badge variant="subtle" colorPalette="blue" maxW="260px" truncate>
                    {chatImage.name}
                  </Badge>
                  <Input
                    value={chatImagePatientId}
                    onChange={(event) => setChatImagePatientId(event.target.value)}
                    placeholder="Patient ID"
                    size="sm"
                    w="160px"
                    autoComplete="off"
                  />
                  <IconButton aria-label="Remove image" size="sm" variant="ghost" onClick={clearChatImage}>
                    <X size={14} />
                  </IconButton>
                </>
              )}
            </HStack>
          </Box>
          <Box
            borderWidth="1px"
            borderColor="gray.200"
            rounded="lg"
            _focusWithin={{ borderColor: "blue.400", boxShadow: "0 0 0 1px var(--chakra-colors-blue-400)" }}
            transition="border-color 0.2s, box-shadow 0.2s"
          >
            <Textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={chatImage ? "Optional note for this upload..." : "Ask about your healthcare data..."}
              border="none"
              _focus={{ boxShadow: "none" }}
              resize="none"
              rows={2}
              fontSize="sm"
              px={3}
              py={2}
            />
            <HStack px={2} py={1.5} justify="space-between">
              <HStack gap={2}>
                <Text fontSize="xs" color="gray.400" display={{ base: "none", sm: "block" }}>
                  {chatImage ? "Image upload will save to Neo4j" : "Enter to send, Shift+Enter for new line"}
                </Text>
              </HStack>
              <Button
                onClick={() => sendMessage()}
                disabled={(!input.trim() && !chatImage) || (chatImage ? !chatImagePatientId.trim() : false) || loading || uploadingImage}
                size="xs"
                colorPalette="blue"
                rounded="md"
                loading={uploadingImage}
              >
                <Send size={14} />
                {chatImage ? "Upload" : ""}
              </Button>
            </HStack>
          </Box>
        </Box>
      </Flex>
    </Flex>
  );
}

function ToolCallTimeline({ toolCalls }: { toolCalls: ToolCall[] }) {
  return (
    <Timeline.Root size="sm" mb={2}>
      {toolCalls.map((tc, j) => (
        <Timeline.Item key={`${tc.name}-${j}`}>
          <Timeline.Connector>
            <Timeline.Separator />
            <Timeline.Indicator
              bg={tc.status === "running" ? "purple.500" : tc.status === "failed" ? "red.500" : "green.500"}
              color="white"
            >
              {tc.status === "running" ? (
                <Spinner size="xs" color="white" />
              ) : (
                <Check size={10} />
              )}
            </Timeline.Indicator>
          </Timeline.Connector>
          <Timeline.Content pb={2}>
            <Collapsible.Root>
              <HStack gap={2}>
                <Badge colorPalette="purple" size="sm">
                  <Wrench size={10} />
                  {tc.name}
                </Badge>
                {tc.status === "running" && (
                  <Text fontSize="xs" color="gray.500">running...</Text>
                )}
                {tc.status === "failed" && (
                  <Text fontSize="xs" color="red.500">failed</Text>
                )}
                {tc.output_preview && (
                  <Collapsible.Trigger asChild>
                    <Button variant="ghost" size="xs" px={1}>
                      <ChevronDown size={12} />
                    </Button>
                  </Collapsible.Trigger>
                )}
              </HStack>
              {tc.output_preview && (
                <Collapsible.Content>
                  <Box
                    mt={1}
                    px={2}
                    py={1}
                    bg="gray.50"
                    borderRadius="sm"
                    fontSize="xs"
                    fontFamily="mono"
                    maxH="120px"
                    overflow="auto"
                  >
                    <Text color="gray.600" mb={1} fontWeight="medium">
                      Inputs: {JSON.stringify(tc.inputs).slice(0, 120)}
                    </Text>
                    <Text color="gray.500">
                      {tc.output_preview.slice(0, 300)}
                      {tc.output_preview.length > 300 && "..."}
                    </Text>
                  </Box>
                </Collapsible.Content>
              )}
            </Collapsible.Root>
          </Timeline.Content>
        </Timeline.Item>
      ))}
    </Timeline.Root>
  );
}

function SkinImageAttachmentView({
  attachment,
  role,
  messageContent,
}: {
  attachment: ChatImageAttachment;
  role: "user" | "assistant" | "system";
  messageContent: string;
}) {
  const [objectUrl, setObjectUrl] = useState("");
  const [loadError, setLoadError] = useState("");

  useEffect(() => {
    let revoked = false;
    let nextObjectUrl = "";

    async function loadImage() {
      const token = getAccessToken();
      if (!token) {
        setLoadError("Please sign in again.");
        return;
      }
      const url = attachment.url.startsWith("http")
        ? attachment.url
        : `${API_BASE.replace(/\/api$/, "")}${attachment.url}`;
      try {
        const response = await fetch(url, {
          headers: { Authorization: `Bearer ${token}` },
        });
        if (!response.ok) throw new Error("Unable to load image");
        const blob = await response.blob();
        nextObjectUrl = URL.createObjectURL(blob);
        if (!revoked) {
          setObjectUrl(nextObjectUrl);
          setLoadError("");
        }
      } catch (err) {
        if (!revoked) {
          setLoadError(err instanceof Error ? err.message : "Unable to load image");
        }
      }
    }

    void loadImage();
    return () => {
      revoked = true;
      if (nextObjectUrl) URL.revokeObjectURL(nextObjectUrl);
    };
  }, [attachment.url]);

  // Use placement if available, otherwise fallback based on role
  const placement = attachment.placement ?? (role === "user" ? "user-upload" : "assistant-result");

  if (placement === "user-upload") {
    return (
      <Box mt={2} maxW="420px" overflow="hidden" borderRadius="xl" borderWidth="1px" borderColor="gray.200">
        {objectUrl ? (
          <Box bg="gray.50">
            <img
              src={objectUrl}
              alt={attachment.title || "Skin image attachment"}
              style={{ width: "100%", maxHeight: "360px", objectFit: "contain", display: "block" }}
            />
          </Box>
        ) : (
          <Flex h="160px" align="center" justify="center" bg="gray.50" color={loadError ? "red.500" : "gray.500"}>
            {loadError ? <Text fontSize="sm">{loadError}</Text> : <Spinner size="sm" />}
          </Flex>
        )}
      </Box>
    );
  }

  // Assistant result view
  const descriptionPreview = attachment.description?.slice(0, 80).toLowerCase() || "";
  const shouldShowDescription = Boolean(
    attachment.description &&
      (!descriptionPreview || !messageContent.toLowerCase().includes(descriptionPreview)),
  );
  const shortDescription = shouldShowDescription && attachment.description
    ? attachment.description.length > 320
      ? `${attachment.description.slice(0, 320).trim()}...`
      : attachment.description
    : "";

  return (
    <Box mt={3}>
      {attachment.title && (
        <Text fontWeight="semibold" fontSize="sm">
          {attachment.title}
        </Text>
      )}

      {attachment.created_at && (
        <Text fontSize="xs" color="gray.500">
          Patient/{attachment.patient_id} · {attachment.created_at}
        </Text>
      )}

      {objectUrl ? (
        <Box bg="gray.50" mt={2} borderRadius="md" overflow="hidden" borderWidth="1px" borderColor="gray.200">
          <img
            src={objectUrl}
            alt={attachment.title || "Skin image attachment"}
            style={{ width: "100%", maxHeight: "360px", objectFit: "contain", display: "block" }}
          />
        </Box>
      ) : (
        <Flex h="160px" align="center" justify="center" bg="gray.50" color={loadError ? "red.500" : "gray.500"} mt={2} borderRadius="md">
          {loadError ? <Text fontSize="sm">{loadError}</Text> : <Spinner size="sm" />}
        </Flex>
      )}

      {shortDescription && (
        <Text mt={2} fontSize="xs" color="gray.600" whiteSpace="pre-wrap" lineClamp={3}>
          {shortDescription}
        </Text>
      )}
    </Box>
  );
}

function SkinDiagnosticResultMessageCard({
  runId,
  result,
}: {
  runId?: string;
  result: SkinDiagnosticResult;
}) {
  return (
    <Box>
      <HStack justify="space-between" align="start" mb={3}>
        <Heading size="xs">Ket qua chan doan da lieu</Heading>
        {runId && <Badge variant="subtle">Run {runId.slice(0, 8)}</Badge>}
      </HStack>
      <VStack align="stretch" gap={3}>
        {result.ranked_diagnoses.map((diagnosis, idx) => {
          const rank = Number(diagnosis.rank || idx + 1);
          const disease = diagnosis.disease?.trim() || "Chua xac dinh";
          const confidence = diagnosis.confidence || "Low";
          const evidenceFor = diagnosis.evidence_for?.trim() || "";
          const evidenceAgainst = diagnosis.evidence_against?.trim() || "";
          return (
            <Box
              key={`${rank}-${disease}`}
              bg="gray.50"
              borderWidth="1px"
              borderColor={diagnosisBorderColor(confidence)}
              borderLeftWidth="5px"
              borderRadius="md"
              p={3}
            >
              <HStack justify="space-between" align="start" gap={3}>
                <Text fontSize="sm" fontWeight="semibold">
                  #{rank} - {confidenceLabel(confidence)}: {disease}
                </Text>
                <Badge colorPalette={confidencePalette(confidence)}>
                  {confidence}
                </Badge>
              </HStack>
              {evidenceFor && (
                <Text fontSize="xs" color="gray.700" mt={2} whiteSpace="pre-wrap">
                  <Text as="span" fontWeight="semibold">Bang chung ho tro: </Text>
                  {evidenceFor}
                </Text>
              )}
              {evidenceAgainst && (
                <Text fontSize="xs" color="gray.700" mt={2} whiteSpace="pre-wrap">
                  <Text as="span" fontWeight="semibold" color="red.700">
                    Diem can loai tru:{" "}
                  </Text>
                  {evidenceAgainst}
                </Text>
              )}
            </Box>
          );
        })}
        {result.reasoning && (
          <Box>
            <Text fontSize="xs" color="gray.500" fontWeight="medium" mb={1}>Bien luan y khoa</Text>
            <Text fontSize="sm" whiteSpace="pre-wrap">{result.reasoning}</Text>
          </Box>
        )}
        {result.visual_observations && (
          <Box>
            <Text fontSize="xs" color="gray.500" fontWeight="medium" mb={1}>Phan tich hinh anh ton thuong</Text>
            <Text fontSize="sm" whiteSpace="pre-wrap">{result.visual_observations}</Text>
          </Box>
        )}
        {result.remaining_uncertainty && (
          <Box borderWidth="1px" borderColor="orange.200" borderRadius="md" bg="orange.50" p={3}>
            <Text fontSize="xs" color="orange.800" fontWeight="medium" mb={1}>
              Thong tin can bo sung
            </Text>
            <Text fontSize="sm" color="orange.900" whiteSpace="pre-wrap">
              {result.remaining_uncertainty}
            </Text>
          </Box>
        )}
        <Text fontSize="xs" color="gray.500">
          Ket qua chi co muc dich ho tro quyet dinh lam sang, khong thay the viec kham truc tiep va chan doan cua bac si.
        </Text>
      </VStack>
    </Box>
  );
}

function ChatSkinDiagnosticRunCard({
  runId,
  onCompleted,
}: {
  runId: string;
  onCompleted?: () => void;
}) {
  const [status, setStatus] = useState<SkinDiagnosticStatus | null>(null);
  const [answers, setAnswers] = useState<Record<number, string>>({});
  const [submittingAnswers, setSubmittingAnswers] = useState(false);
  const [error, setError] = useState("");
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const submittedStepRef = useRef<string | null>(null);
  const completedNotifiedRef = useRef(false);
  const onCompletedRef = useRef(onCompleted);

  useEffect(() => {
    onCompletedRef.current = onCompleted;
  }, [onCompleted]);

  const stopPolling = useCallback(() => {
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }, []);

  const pollStatus = useCallback(async () => {
    try {
      const next = await getSkinDiagnosticStatus(runId);
      if (next.status === "interrupt" && submittedStepRef.current === next.current_step) {
        setStatus((prev) => prev ? { ...prev, status: "running", pending_questions: null } : next);
        return;
      }
      if (next.status !== "interrupt" || submittedStepRef.current !== next.current_step) {
        submittedStepRef.current = null;
      }
      setStatus(next);
      setError("");
      if (next.status === "completed" || next.status === "error" || next.status === "interrupt") {
        stopPolling();
      }
      if (next.status === "completed" && !completedNotifiedRef.current) {
        completedNotifiedRef.current = true;
        window.setTimeout(() => onCompletedRef.current?.(), 500);
      }
    } catch (err) {
      stopPolling();
      setError(err instanceof Error ? err.message : "Unable to load diagnostic status");
    }
  }, [runId, stopPolling]);

  const startPolling = useCallback(() => {
    stopPolling();
    void pollStatus();
    pollRef.current = setInterval(() => pollStatus(), 2000);
  }, [pollStatus, stopPolling]);

  useEffect(() => {
    startPolling();
    return stopPolling;
  }, [startPolling, stopPolling]);

  function setAnswer(question: SkinPendingQuestion, answer: string) {
    const questionNum = question.question_num;
    if (questionNum === null) return;
    setAnswers((prev) => ({ ...prev, [questionNum]: answer }));
  }

  async function submitAnswers() {
    if (status?.status !== "interrupt" || !status.pending_questions || submittingAnswers) return;
    const submittedStep = status.current_step;
    const payload = status.pending_questions.map((question) => ({
      question_num: question.question_num,
      answer: answers[question.question_num ?? -1],
    }));
    if (payload.some((item) => !item.answer)) {
      setError("Answer every question before submitting.");
      return;
    }

    setSubmittingAnswers(true);
    setError("");
    submittedStepRef.current = submittedStep;
    try {
      await submitSkinDiagnosticAnswers(runId, payload);
      setAnswers({});
      setStatus((prev) => prev ? { ...prev, status: "running", pending_questions: null } : prev);
      startPolling();
    } catch (err) {
      submittedStepRef.current = null;
      setError(err instanceof Error ? err.message : "Unable to submit answers");
      try {
        const latest = await getSkinDiagnosticStatus(runId);
        setStatus(latest);
      } catch {
        // Keep the submit error visible.
      }
    } finally {
      setSubmittingAnswers(false);
    }
  }

  const pendingQuestions = status?.status === "interrupt" ? status.pending_questions || [] : [];
  const allAnswered = pendingQuestions.length > 0 &&
    pendingQuestions.every((question) => answers[question.question_num ?? -1]);
  const displayStep = status?.current_step || "visual_extract";
  const displayProgress = status?.progress ?? 1;

  return (
    <Box mt={3} borderWidth="1px" borderColor="gray.200" borderRadius="md" overflow="hidden" bg="white">
      <HStack px={3} py={2} justify="space-between" borderBottom="1px solid" borderColor="gray.100">
        <Box minW={0}>
          <Text fontSize="sm" fontWeight="semibold">Skin diagnostic run</Text>
          <Text fontSize="xs" color="gray.500" truncate>{runId}</Text>
        </Box>
        <HStack gap={2}>
          {(!status || status.status === "running") && <Spinner size="xs" />}
          <Badge colorPalette={
            status?.status === "error" ? "red" :
            status?.status === "completed" ? "green" :
            status?.status === "interrupt" ? "orange" :
            "blue"
          }>
            {status?.status || "running"}
          </Badge>
        </HStack>
      </HStack>

      <Box p={3}>
        {error && (
          <Box mb={3} px={3} py={2} bg="red.50" color="red.700" fontSize="sm" borderRadius="md">
            {error}
          </Box>
        )}

        <Box mb={3}>
          <HStack justify="space-between" mb={2}>
            <Text fontSize="sm" fontWeight="medium">{stepLabel(displayStep)}</Text>
            <Text fontSize="xs" color="gray.500">{Math.min(displayProgress, 7)}/7</Text>
          </HStack>
          <Box h="2" bg="gray.100" borderRadius="full" overflow="hidden">
            <Box
              h="100%"
              bg="blue.500"
              width={`${Math.min((displayProgress / 7) * 100, 100)}%`}
              transition="width 0.2s"
            />
          </Box>
        </Box>

        {pendingQuestions.length > 0 && (
          <Box>
            <Heading size="xs" mb={3}>Clinical questions</Heading>
            <VStack align="stretch" gap={3}>
              {pendingQuestions.map((question) => {
                const qNum = question.question_num ?? -1;
                return (
                  <Box key={qNum} pb={3} borderBottom="1px solid" borderColor="gray.100">
                    <HStack align="start" justify="space-between" gap={3}>
                      <Text fontSize="sm" fontWeight="medium">
                        {question.question_num}. {question.question}
                      </Text>
                      {question.pqrst_category && <Badge size="sm">{question.pqrst_category}</Badge>}
                    </HStack>
                    {question.purpose && (
                      <Text fontSize="xs" color="gray.500" mt={1}>{question.purpose}</Text>
                    )}
                    <HStack gap={2} mt={2}>
                      <Button
                        size="xs"
                        variant={answers[qNum] === "yes" ? "solid" : "outline"}
                        colorPalette="green"
                        onClick={() => setAnswer(question, "yes")}
                      >
                        Yes
                      </Button>
                      <Button
                        size="xs"
                        variant={answers[qNum] === "no" ? "solid" : "outline"}
                        colorPalette="red"
                        onClick={() => setAnswer(question, "no")}
                      >
                        No
                      </Button>
                    </HStack>
                  </Box>
                );
              })}
            </VStack>
            <Button mt={3} size="sm" colorPalette="blue" disabled={!allAnswered} loading={submittingAnswers} onClick={submitAnswers}>
              <Check size={14} />
              Submit answers
            </Button>
          </Box>
        )}

        {hasDiagnosticResult(status) && (
          <SkinDiagnosticResultMessageCard runId={runId} result={status.result} />
        )}
      </Box>
    </Box>
  );
}
