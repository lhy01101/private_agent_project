package com.fangtang.ai.viewmodel

import android.graphics.Bitmap
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.fangtang.ai.api.ApiClient
import com.fangtang.ai.model.ChatMessage
import com.fangtang.ai.model.Conversation
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import okhttp3.sse.EventSource

class ChatViewModel : ViewModel() {

    var conversations by mutableStateOf<List<Conversation>>(emptyList())
        private set

    var currentConvId by mutableStateOf<String?>(null)
        private set

    var messages by mutableStateOf<List<ChatMessage>>(emptyList())
        private set

    var isStreaming by mutableStateOf(false)
        private set

    var toolStatus by mutableStateOf<String?>(null)
        private set

    var errorMessage by mutableStateOf<String?>(null)
        private set

    var fontSizeScale by mutableStateOf(1.0f)
        private set

    var isDarkTheme by mutableStateOf(true)
        private set

    var backgroundImage by mutableStateOf<Bitmap?>(null)
        private set

    private var eventSource: EventSource? = null

    init {
        loadConversations()
    }

    fun loadConversations() {
        viewModelScope.launch {
            try {
                conversations = withContext(Dispatchers.IO) {
                    ApiClient.listConversations()
                }
            } catch (e: Exception) {
                errorMessage = e.message
            }
        }
    }

    fun createAndSelectConversation() {
        viewModelScope.launch {
            try {
                val conv = withContext(Dispatchers.IO) {
                    ApiClient.createConversation()
                }
                conversations = conversations + conv
                selectConversation(conv.id)
            } catch (e: Exception) {
                errorMessage = e.message
            }
        }
    }

    fun selectConversation(convId: String) {
        currentConvId = convId
        messages = emptyList()
        viewModelScope.launch {
            try {
                val history = withContext(Dispatchers.IO) {
                    ApiClient.getHistory(convId)
                }
                messages = history
            } catch (e: Exception) {
                errorMessage = e.message
            }
        }
    }

    fun deleteConversation(convId: String) {
        viewModelScope.launch {
            try {
                withContext(Dispatchers.IO) {
                    ApiClient.deleteConversation(convId)
                }
                conversations = conversations.filter { it.id != convId }
                if (currentConvId == convId) {
                    currentConvId = null
                    messages = emptyList()
                }
            } catch (e: Exception) {
                errorMessage = e.message
            }
        }
    }

    fun sendMessage(text: String) {
        val convId = currentConvId ?: return
        if (text.isBlank() || isStreaming) return

        messages = messages + ChatMessage("user", text)
        val assistantIndex = messages.size
        messages = messages + ChatMessage("assistant", "", isStreaming = true)
        isStreaming = true
        errorMessage = null

        eventSource = ApiClient.streamChat(
            convId = convId,
            message = text,
            onText = { chunk ->
                val current = messages[assistantIndex]
                messages = messages.toMutableList().apply {
                    set(assistantIndex, current.copy(content = current.content + chunk))
                }
            },
            onTool = { status ->
                toolStatus = status
            },
            onToolEnd = {
                toolStatus = null
            },
            onError = { error ->
                isStreaming = false
                errorMessage = error
                val current = messages[assistantIndex]
                if (current.content.isEmpty()) {
                    messages = messages.toMutableList().apply {
                        set(assistantIndex, current.copy(content = "Error: $error", isStreaming = false))
                    }
                }
            },
            onDone = {
                isStreaming = false
                val current = messages[assistantIndex]
                messages = messages.toMutableList().apply {
                    set(assistantIndex, current.copy(isStreaming = false))
                }
                loadConversations()
            },
        )
    }

    fun goBackToList() {
        eventSource?.cancel()
        eventSource = null
        isStreaming = false
        currentConvId = null
        messages = emptyList()
        loadConversations()
    }

    fun updateServerUrl(url: String) {
        ApiClient.baseUrl = url.trimEnd('/')
    }

    fun updateFontSize(scale: Float) {
        fontSizeScale = scale
    }

    fun toggleTheme() {
        isDarkTheme = !isDarkTheme
    }

    fun updateBackgroundImage(bitmap: Bitmap?) {
        backgroundImage = bitmap
    }

    fun renameConversation(convId: String, newTitle: String) {
        viewModelScope.launch {
            try {
                withContext(Dispatchers.IO) {
                    ApiClient.updateTitle(convId, newTitle)
                }
                conversations = conversations.map {
                    if (it.id == convId) it.copy(title = newTitle) else it
                }
            } catch (e: Exception) {
                errorMessage = e.message
            }
        }
    }

    fun clearError() {
        errorMessage = null
    }

    override fun onCleared() {
        super.onCleared()
        eventSource?.cancel()
    }
}
