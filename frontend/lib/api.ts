"use client";

import { API_BASE } from "@/lib/config";

const ACCESS_TOKEN_KEY = "fhir-agent-access-token";
const USER_KEY = "fhir-agent-user";

export interface UserProfile {
  id: string;
  username: string;
  external_id: string | null;
  created_at: string;
  updated_at: string;
}

export interface TokenResponse {
  access_token: string;
  token_type: string;
  expires_in: number;
  user: UserProfile;
}

export interface Conversation {
  id: string;
  user_id: string;
  title: string;
  created_at: string;
  updated_at: string;
}

export interface ChatMessage {
  id: string;
  conversation_id: string;
  role: "user" | "assistant" | "system";
  content: string;
  created_at: string;
}

export interface ConversationListResponse {
  items: Conversation[];
  total: number;
}

export interface MessageListResponse {
  items: ChatMessage[];
  total: number;
}

export interface InitialExchangeResponse {
  conversation: Conversation;
  user_message: ChatMessage;
  assistant_message: ChatMessage;
}

export interface MessageExchangeResponse {
  conversation_id: string;
  user_message: ChatMessage;
  assistant_message: ChatMessage;
}

export class ApiError extends Error {
  status: number;
  detail: string;

  constructor(status: number, detail: string) {
    super(detail);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

export function getAccessToken(): string | null {
  if (typeof window === "undefined") return null;
  return localStorage.getItem(ACCESS_TOKEN_KEY);
}

export function getStoredUser(): UserProfile | null {
  if (typeof window === "undefined") return null;
  const raw = localStorage.getItem(USER_KEY);
  if (!raw) return null;
  try {
    return JSON.parse(raw) as UserProfile;
  } catch {
    return null;
  }
}

export function storeAuth(tokenResponse: TokenResponse): void {
  localStorage.setItem(ACCESS_TOKEN_KEY, tokenResponse.access_token);
  localStorage.setItem(USER_KEY, JSON.stringify(tokenResponse.user));
}

export function clearAuth(): void {
  localStorage.removeItem(ACCESS_TOKEN_KEY);
  localStorage.removeItem(USER_KEY);
}

async function readError(response: Response): Promise<string> {
  const fallback = `Backend error (${response.status})`;
  const data = await response.json().catch(() => null);
  if (!data || typeof data !== "object" || !("detail" in data)) return fallback;
  const detail = data.detail;
  if (typeof detail === "string") return detail;
  return fallback;
}

export async function apiFetch(path: string, init: RequestInit = {}): Promise<Response> {
  const token = getAccessToken();
  const headers = new Headers(init.headers);
  if (!headers.has("Accept")) headers.set("Accept", "application/json");
  if (token) headers.set("Authorization", `Bearer ${token}`);

  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers,
  });

  if (response.status === 401) {
    clearAuth();
    throw new ApiError(401, "Please sign in again.");
  }

  return response;
}

async function jsonRequest<T>(path: string, init: RequestInit = {}): Promise<T> {
  const response = await apiFetch(path, init);
  if (!response.ok) {
    throw new ApiError(response.status, await readError(response));
  }
  return response.json() as Promise<T>;
}

export async function login(username: string, password: string): Promise<TokenResponse> {
  const response = await fetch(`${API_BASE}/auth/login`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "application/json",
    },
    body: JSON.stringify({ username, password }),
  });
  if (!response.ok) {
    throw new ApiError(response.status, await readError(response));
  }
  const tokenResponse = (await response.json()) as TokenResponse;
  storeAuth(tokenResponse);
  return tokenResponse;
}

export async function register(username: string, password: string): Promise<UserProfile> {
  const response = await fetch(`${API_BASE}/auth/register`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "application/json",
    },
    body: JSON.stringify({ username, password }),
  });
  if (!response.ok) {
    throw new ApiError(response.status, await readError(response));
  }
  return response.json() as Promise<UserProfile>;
}

export async function listConversations(skip = 0, limit = 50): Promise<ConversationListResponse> {
  return jsonRequest<ConversationListResponse>(`/conversations?skip=${skip}&limit=${limit}`);
}

export async function listMessages(
  conversationId: string,
  skip = 0,
  limit = 200,
): Promise<MessageListResponse> {
  return jsonRequest<MessageListResponse>(
    `/conversations/${conversationId}/messages?skip=${skip}&limit=${limit}`,
  );
}

export async function deleteConversation(conversationId: string): Promise<void> {
  const response = await apiFetch(`/conversations/${conversationId}`, {
    method: "DELETE",
  });
  if (response.status === 204) return;
  if (!response.ok) {
    throw new ApiError(response.status, await readError(response));
  }
}

export async function createConversation(firstMessage: string): Promise<InitialExchangeResponse> {
  return jsonRequest<InitialExchangeResponse>("/conversations", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ first_message: firstMessage }),
  });
}

export async function sendConversationMessage(
  conversationId: string,
  content: string,
): Promise<MessageExchangeResponse> {
  return jsonRequest<MessageExchangeResponse>(`/conversations/${conversationId}/messages`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ content }),
  });
}

export async function openConversationStream(firstMessage: string, signal: AbortSignal): Promise<Response> {
  return apiFetch("/conversations/stream", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "text/event-stream",
    },
    body: JSON.stringify({ first_message: firstMessage }),
    signal,
  });
}

export async function openMessageStream(
  conversationId: string,
  content: string,
  signal: AbortSignal,
): Promise<Response> {
  return apiFetch(`/conversations/${conversationId}/messages/stream`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "text/event-stream",
    },
    body: JSON.stringify({ content }),
    signal,
  });
}
