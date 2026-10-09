package com.fangtang.ai.model

data class Conversation(
    val id: String,
    val title: String,
    val created_at: String,
)

data class ChatMessage(
    val role: String,
    val content: String,
    val isStreaming: Boolean = false,
)
