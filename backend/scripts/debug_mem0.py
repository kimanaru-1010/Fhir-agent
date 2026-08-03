"""Debug save_conversation_memory with print statements."""

import asyncio
import sys

from app.services.long_term_memory import init_memory, get_memory, save_conversation_memory, _memory, _sanitize, _strip_reasoning


async def main():
    print(f"BEFORE init: _memory = {_memory}", file=sys.stderr)

    ok = await init_memory()
    print(f"init_memory returned: {ok}", file=sys.stderr)

    mem = get_memory()
    print(f"get_memory() = {type(mem).__name__ if mem else None}", file=sys.stderr)

    # Test _sanitize
    test_msg = "The user prefers to receive answers in Vietnamese language"
    cleaned = _sanitize(test_msg)
    print(f"_sanitize('{test_msg[:30]}...') = '{cleaned[:30]}...' (len={len(cleaned)})", file=sys.stderr)

    test_msg2 = "I understand. I'll answer in Vietnamese."
    cleaned2 = _strip_reasoning(_sanitize(test_msg2))
    print(f"_strip_reasoning(_sanitize(...)) = '{cleaned2[:30]}...' (len={len(cleaned2)})", file=sys.stderr)

    print("\n=== Calling save_conversation_memory ===", file=sys.stderr)
    result = await save_conversation_memory(
        user_id="smoke-test-user",
        session_id="smoke-test-session",
        user_message="The user prefers to receive answers in Vietnamese language",
        assistant_message="I understand. I'll answer in Vietnamese.",
    )
    print(f"save_conversation_memory result: {result}", file=sys.stderr)


asyncio.run(main())
