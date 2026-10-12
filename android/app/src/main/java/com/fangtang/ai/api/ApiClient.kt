package com.fangtang.ai.api

import com.fangtang.ai.model.Conversation
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.MultipartBody
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import okhttp3.Response
import okhttp3.sse.EventSource
import okhttp3.sse.EventSourceListener
import okhttp3.sse.EventSources
import java.util.concurrent.TimeUnit

object ApiClient {

    var baseUrl: String = "http://192.168.1.4:8000"

    private val json = Json { ignoreUnknownKeys = true }
    private val client = OkHttpClient.Builder()
        .connectTimeout(30, TimeUnit.SECONDS)
        .readTimeout(0, TimeUnit.MINUTES)
        .writeTimeout(30, TimeUnit.SECONDS)
        .build()

    private val JSON_MEDIA = "application/json".toMediaType()

    suspend fun listConversations(): List<Conversation> {
        val request = Request.Builder()
            .url("$baseUrl/api/conversations")
            .get()
            .build()
        val response = client.newCall(request).execute()
        val body = response.body?.string() ?: return emptyList()
        return json.decodeFromString<List<ConversationDto>>(body)
            .map { Conversation(it.id, it.title, it.created_at) }
    }

    suspend fun createConversation(): Conversation {
        val request = Request.Builder()
            .url("$baseUrl/api/conversations")
            .post("".toRequestBody(JSON_MEDIA))
            .build()
        val response = client.newCall(request).execute()
        val body = response.body?.string() ?: throw Exception("Empty response")
        val dto = json.decodeFromString<ConversationDto>(body)
        return Conversation(dto.id, dto.title, dto.created_at)
    }

    suspend fun deleteConversation(convId: String) {
        val request = Request.Builder()
            .url("$baseUrl/api/conversations/$convId")
            .delete()
            .build()
        client.newCall(request).execute()
    }

    suspend fun updateTitle(convId: String, title: String) {
        val body = """{"title":"${title.replace("\"", "\\\"")}"}"""
        val request = Request.Builder()
            .url("$baseUrl/api/conversations/$convId/title")
            .post(body.toRequestBody(JSON_MEDIA))
            .build()
        client.newCall(request).execute()
    }

    suspend fun getHistory(convId: String): List<com.fangtang.ai.model.ChatMessage> {
        val request = Request.Builder()
            .url("$baseUrl/api/history/$convId")
            .get()
            .build()
        val response = client.newCall(request).execute()
        val body = response.body?.string() ?: return emptyList()
        val historyDto = json.decodeFromString<HistoryDto>(body)
        return historyDto.messages.map {
            com.fangtang.ai.model.ChatMessage(it.role, it.content)
        }
    }

    suspend fun uploadFile(filename: String, bytes: ByteArray): com.fangtang.ai.model.PendingFile {
        val body = MultipartBody.Builder()
            .setType(MultipartBody.FORM)
            .addFormDataPart(
                "file",
                filename,
                bytes.toRequestBody("application/octet-stream".toMediaTypeOrNull()),
            )
            .build()
        val request = Request.Builder()
            .url("$baseUrl/api/upload")
            .post(body)
            .build()
        val response = client.newCall(request).execute()
        if (!response.isSuccessful) {
            val err = response.body?.string() ?: "Upload failed"
            throw Exception(err)
        }
        val respBody = response.body?.string() ?: throw Exception("Empty response")
        val dto = json.decodeFromString<UploadResponseDto>(respBody)
        return com.fangtang.ai.model.PendingFile(dto.filename, dto.content)
    }

    fun streamChat(
        convId: String,
        message: String,
        files: List<com.fangtang.ai.model.PendingFile> = emptyList(), // [NEW]
        onText: (String) -> Unit,
        onTool: (String) -> Unit,
        onToolEnd: () -> Unit,
        onError: (String) -> Unit,
        onDone: () -> Unit,
    ): EventSource {
        // [NEW] 构建包含 files 的 JSON body
        val filesJson = if (files.isEmpty()) "" else {
            val arr = files.joinToString(",") { f ->
                """{"filename":"${f.filename.replace("\"", "\\\"")}","content":"${f.content.replace("\\", "\\\\").replace("\"", "\\\"").replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")}"}"""
            }
            ""","files":[$arr]"""
        }
        val body = """{"message":"${message.replace("\"", "\\\"")}"$filesJson}"""
        val request = Request.Builder()
            .url("$baseUrl/api/chat/$convId")
            .post(body.toRequestBody(JSON_MEDIA))
            .header("Accept", "text/event-stream")
            .build()

        val factory = EventSources.createFactory(client)

        return factory.newEventSource(request, object : EventSourceListener() {
            override fun onEvent(eventSource: EventSource, id: String?, type: String?, data: String) {
                when (type) {
                    "text" -> {
                        val content = json.decodeFromString<TextDto>(data).content
                        onText(content)
                    }
                    "tool" -> {
                        val content = json.decodeFromString<TextDto>(data).content
                        onTool(content)
                    }
                    "tool_end" -> onToolEnd()
                    "error" -> {
                        val content = json.decodeFromString<TextDto>(data).content
                        onError(content)
                    }
                    "done" -> onDone()
                }
            }

            override fun onFailure(eventSource: EventSource, t: Throwable?, response: Response?) {
                onError(t?.message ?: "Connection failed")
            }
        })
    }

    @Serializable
    private data class ConversationDto(
        val id: String,
        val title: String,
        val created_at: String,
    )

    @Serializable
    private data class HistoryDto(
        val messages: List<MessageDto>,
    )

    @Serializable
    private data class MessageDto(
        val role: String,
        val content: String,
    )

    @Serializable
    private data class TextDto(
        val content: String,
    )

    @Serializable
    private data class UploadResponseDto(
        val filename: String,
        val content: String,
    )
}
