"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  Badge,
  Box,
  Button,
  Flex,
  Heading,
  HStack,
  Input,
  Separator,
  Spinner,
  Text,
  Textarea,
  VStack,
} from "@chakra-ui/react";
import { Check, ImagePlus, Send, Stethoscope } from "lucide-react";
import {
  ApiError,
  getAccessToken,
  getSkinDiagnosticStatus,
  startSkinDiagnostic,
  submitSkinDiagnosticAnswers,
} from "@/lib/api";
import type {
  SkinDiagnosticResult,
  SkinDiagnosticStatus,
  SkinPendingQuestion,
} from "@/lib/api";


function hasResult(status: SkinDiagnosticStatus | null): status is SkinDiagnosticStatus & { result: SkinDiagnosticResult } {
  return Boolean(
    status &&
    status.status === "completed" &&
    "ranked_diagnoses" in status.result,
  );
}

function stepLabel(step: string): string {
  const labels: Record<string, string> = {
    visual_extract: "Image analysis",
    knowledge_base: "Knowledge base search",
    clinical_planner_round1: "Question planning 1",
    user_interview_round1: "Interview 1",
    clinical_planner_round2: "Question planning 2",
    user_interview_round2: "Interview 2",
    diagnostic_reasoning: "Diagnostic reasoning",
  };
  return labels[step] || step || "Waiting";
}

export function SkinDiagnosticPanel() {
  const [image, setImage] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [anamnesis, setAnamnesis] = useState("");
  const [runId, setRunId] = useState<string | null>(null);
  const [status, setStatus] = useState<SkinDiagnosticStatus | null>(null);
  const [answers, setAnswers] = useState<Record<number, string>>({});
  const [loading, setLoading] = useState(false);
  const [submittingAnswers, setSubmittingAnswers] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const submittedStepRef = useRef<string | null>(null);

  const stopPolling = useCallback(() => {
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }, []);

  const pollStatus = useCallback(async (id: string) => {
    try {
      const next = await getSkinDiagnosticStatus(id);
      if (next.status === "interrupt" && submittedStepRef.current === next.current_step) {
        setStatus((prev) => prev ? { ...prev, status: "running", pending_questions: null } : next);
        return;
      }
      if (next.status !== "interrupt" || submittedStepRef.current !== next.current_step) {
        submittedStepRef.current = null;
      }
      setStatus(next);
      if (next.status === "completed" || next.status === "error" || next.status === "interrupt") {
        stopPolling();
      }
    } catch (err) {
      stopPolling();
      setError(err instanceof Error ? err.message : "Unable to load diagnostic status");
    }
  }, [stopPolling]);

  const startPolling = useCallback((id: string) => {
    stopPolling();
    void pollStatus(id);
    pollRef.current = setInterval(() => pollStatus(id), 2000);
  }, [pollStatus, stopPolling]);

  useEffect(() => () => {
    stopPolling();
    if (previewUrl) URL.revokeObjectURL(previewUrl);
  }, [previewUrl, stopPolling]);

  function onImageChange(file: File | null) {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setImage(file);
    setPreviewUrl(file ? URL.createObjectURL(file) : "");
  }

  async function startRun() {
    if (!image || loading) return;
    if (!getAccessToken()) {
      setError("Sign in from the Chat tab before using skin diagnostics.");
      return;
    }
    setLoading(true);
    setError(null);
    setStatus(null);
    setAnswers({});
    submittedStepRef.current = null;
    stopPolling();
    try {
      const started = await startSkinDiagnostic(image, anamnesis);
      setRunId(started.run_id);
      setStatus({
        run_id: started.run_id,
        status: "running",
        current_step: started.current_step,
        progress: 1,
        pending_questions: null,
        result: {},
        error: null,
      });
      startPolling(started.run_id);
    } catch (err) {
      if (err instanceof ApiError && err.status === 401) {
        setError("Please sign in again.");
      } else {
        setError(err instanceof Error ? err.message : "Unable to start diagnostic run");
      }
    } finally {
      setLoading(false);
    }
  }

  function setAnswer(question: SkinPendingQuestion, answer: string) {
    const questionNum = question.question_num;
    if (questionNum === null) return;
    setAnswers((prev) => ({ ...prev, [questionNum]: answer }));
  }

  async function submitAnswers() {
    if (!runId || status?.status !== "interrupt" || !status.pending_questions || submittingAnswers) return;
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
    setError(null);
    submittedStepRef.current = submittedStep;
    try {
      await submitSkinDiagnosticAnswers(runId, payload);
      setAnswers({});
      setStatus((prev) => prev ? { ...prev, status: "running", pending_questions: null } : prev);
      startPolling(runId);
    } catch (err) {
      if (err instanceof ApiError && err.status === 400 && err.detail.includes("not waiting")) {
        const latest = await getSkinDiagnosticStatus(runId);
        submittedStepRef.current = null;
        setStatus(latest);
        setAnswers({});
        if (latest.status !== "completed") {
          setError(err.detail);
        }
      } else {
        submittedStepRef.current = null;
        setError(err instanceof Error ? err.message : "Unable to submit answers");
      }
    } finally {
      setSubmittingAnswers(false);
    }
  }

  const pendingQuestions = status?.status === "interrupt" ? status.pending_questions || [] : [];
  const allAnswered = pendingQuestions.length > 0 &&
    pendingQuestions.every((question) => answers[question.question_num ?? -1]);

  return (
    <Flex h="100%" direction="column">
      <HStack px={4} py={3} borderBottom="1px solid" borderColor="gray.200" justify="space-between">
        <HStack gap={2}>
          <Stethoscope size={18} />
          <Heading size="sm">Skin Diagnostic</Heading>
        </HStack>
        {status && (
          <Badge colorPalette={status.status === "error" ? "red" : status.status === "completed" ? "green" : "blue"}>
            {status.status}
          </Badge>
        )}
      </HStack>

      {error && (
        <Box px={4} py={2} bg="red.50" color="red.700" fontSize="sm">
          {error}
        </Box>
      )}

      <Flex flex={1} minH={0} overflow="auto" px={4} py={4} gap={4} direction={{ base: "column", lg: "row" }}>
        <VStack align="stretch" gap={3} flex="0 0 340px">
          <Box>
            <Text fontSize="xs" fontWeight="medium" color="gray.600" mb={1}>Image</Text>
            <Input
              type="file"
              accept="image/png,image/jpeg,image/webp"
              onChange={(event) => onImageChange(event.target.files?.[0] ?? null)}
            />
          </Box>
          {previewUrl ? (
            <Box overflow="hidden" borderWidth="1px" borderColor="gray.200" borderRadius="md" bg="gray.50">
              <img src={previewUrl} alt="Selected lesion" style={{ width: "100%", display: "block" }} />
            </Box>
          ) : (
            <Flex h="180px" align="center" justify="center" borderWidth="1px" borderColor="gray.200" borderRadius="md" color="gray.400">
              <VStack gap={1}>
                <ImagePlus size={24} />
                <Text fontSize="sm">No image selected</Text>
              </VStack>
            </Flex>
          )}
          <Box>
            <Text fontSize="xs" fontWeight="medium" color="gray.600" mb={1}>Initial complaint</Text>
            <Textarea
              value={anamnesis}
              onChange={(event) => setAnamnesis(event.target.value)}
              placeholder="Describe symptoms, location, duration, or triggers..."
              rows={5}
              resize="vertical"
            />
          </Box>
          <Button colorPalette="blue" onClick={startRun} disabled={!image || loading} loading={loading}>
            <Send size={14} />
            Start diagnostic run
          </Button>
        </VStack>

        <VStack align="stretch" gap={4} flex={1} minW={0}>
          {status ? (
            <Box borderWidth="1px" borderColor="gray.200" borderRadius="md" p={3}>
              <HStack justify="space-between" mb={2}>
                <Text fontSize="sm" fontWeight="medium">{stepLabel(status.current_step)}</Text>
                {(status.status === "running" || loading) && <Spinner size="xs" />}
              </HStack>
              <Box h="2" bg="gray.100" borderRadius="full" overflow="hidden">
                <Box
                  h="100%"
                  bg="blue.500"
                  width={`${Math.min((status.progress / 7) * 100, 100)}%`}
                  transition="width 0.2s"
                />
              </Box>
            </Box>
          ) : (
            <Flex flex={1} align="center" justify="center" minH="260px" color="gray.500">
              <Text fontSize="sm">Upload an image and start a run.</Text>
            </Flex>
          )}

          {pendingQuestions.length > 0 && (
            <Box borderWidth="1px" borderColor="gray.200" borderRadius="md" p={3}>
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

          {hasResult(status) && (
            <Box borderWidth="1px" borderColor="gray.200" borderRadius="md" p={3}>
              <Heading size="xs" mb={3}>Diagnostic result</Heading>
              <VStack align="stretch" gap={3}>
                {status.result.ranked_diagnoses.map((diagnosis, idx) => {
                  const disease = String(diagnosis.disease || "Unspecified");
                  const likelihood = diagnosis.likelihood ? String(diagnosis.likelihood) : "";
                  const evidence = Array.isArray(diagnosis.supporting_evidence)
                    ? diagnosis.supporting_evidence.map(String).join("; ")
                    : "";
                  return (
                    <Box key={`${disease}-${idx}`} bg="gray.50" borderRadius="md" p={3}>
                      <HStack justify="space-between" align="start">
                        <Text fontSize="sm" fontWeight="semibold">
                          {idx + 1}. {disease}
                        </Text>
                        {likelihood && <Badge colorPalette="purple">{likelihood}</Badge>}
                      </HStack>
                      {evidence && (
                        <Text fontSize="xs" color="gray.600" mt={2}>
                          Evidence: {evidence}
                        </Text>
                      )}
                    </Box>
                  );
                })}
                <Separator />
                <Box>
                  <Text fontSize="xs" color="gray.500" fontWeight="medium" mb={1}>Reasoning</Text>
                  <Text fontSize="sm" whiteSpace="pre-wrap">{status.result.reasoning}</Text>
                </Box>
                <Box>
                  <Text fontSize="xs" color="gray.500" fontWeight="medium" mb={1}>Visual observations</Text>
                  <Text fontSize="sm" whiteSpace="pre-wrap">{status.result.visual_observations}</Text>
                </Box>
              </VStack>
            </Box>
          )}
        </VStack>
      </Flex>
    </Flex>
  );
}
