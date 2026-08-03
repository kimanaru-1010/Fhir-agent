"use client";

import { useEffect, useState } from "react";
import { Box, Button, Flex, Heading, HStack, Text } from "@chakra-ui/react";
import { ChatInterface } from "@/components/ChatInterface";
import { SkinDiagnosticPanel } from "@/components/SkinDiagnosticPanel";
import { API_BASE, DOMAIN } from "@/lib/config";

export default function Home() {
  const [backendStatus, setBackendStatus] = useState<"ok" | "degraded" | "offline">("offline");
  const [mode, setMode] = useState<"chat" | "skin">("chat");

  useEffect(() => {
    async function checkHealth(retries = 3, delay = 1000) {
      for (let attempt = 0; attempt < retries; attempt++) {
        try {
          const res = await fetch(`${API_BASE.replace("/api", "")}/health`, {
            signal: AbortSignal.timeout(5000),
          });
          const data = await res.json();
          setBackendStatus(data.status === "ok" ? "ok" : "degraded");
          return;
        } catch {
          if (attempt < retries - 1) {
            await new Promise((resolve) => setTimeout(resolve, delay * (attempt + 1)));
          }
        }
      }
      setBackendStatus("offline");
    }

    checkHealth();
    const interval = setInterval(() => checkHealth(1), 60000);
    return () => clearInterval(interval);
  }, []);

  return (
    <Flex direction="column" h="100dvh" bg="white">
      <Flex bg="gray.900" color="white" px={6} py={3} justify="space-between" align="center">
        <Box>
          <Heading size="md">{DOMAIN.name} Assistant</Heading>
          <Text fontSize="sm" color="gray.400">
            {DOMAIN.tagline}
          </Text>
        </Box>
        <HStack gap={2}>
          <HStack gap={1} bg="gray.800" p={1} borderRadius="md">
            <Button
              size="xs"
              variant={mode === "chat" ? "solid" : "ghost"}
              colorPalette={mode === "chat" ? "blue" : "gray"}
              onClick={() => setMode("chat")}
            >
              Chat
            </Button>
            <Button
              size="xs"
              variant={mode === "skin" ? "solid" : "ghost"}
              colorPalette={mode === "skin" ? "blue" : "gray"}
              onClick={() => setMode("skin")}
            >
              Skin
            </Button>
          </HStack>
          <Box
            w={3}
            h={3}
            borderRadius="full"
            bg={
              backendStatus === "ok"
                ? "green.400"
                : backendStatus === "degraded"
                  ? "yellow.400"
                  : "red.400"
            }
            title={
              backendStatus === "ok"
                ? "Backend connected"
                : backendStatus === "degraded"
                  ? "Backend connected with limited services"
                  : "Backend offline"
            }
          />
          <Text fontSize="xs" color="gray.500">
            {backendStatus === "ok"
              ? "Connected"
              : backendStatus === "degraded"
                ? "Degraded"
                : "Offline"}
          </Text>
        </HStack>
      </Flex>

      <Box as="main" flex={1} minH={0} overflow="hidden">
        {mode === "chat" ? <ChatInterface /> : <SkinDiagnosticPanel />}
      </Box>
    </Flex>
  );
}
