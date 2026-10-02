const API_BASE = '';

let conversations = [];
let activeConvId = null;
let isStreaming = false;
const messageCache = new Map();

const messagesEl = document.getElementById('messages');
const welcomeEl = document.getElementById('welcome');
const inputEl = document.getElementById('message-input');
const sendBtn = document.getElementById('send-btn');
const newBtn = document.getElementById('new-chat-btn');
const convListEl = document.getElementById('conversation-list');
const titleEl = document.getElementById('current-title');
const sidebarEl = document.getElementById('sidebar');
const sidebarToggle = document.getElementById('sidebar-toggle');
const chatContainer = document.getElementById('chat-container');

marked.setOptions({
    highlight(code, lang) {
        if (lang && hljs.getLanguage(lang)) {
            return hljs.highlight(code, { language: lang }).value;
        }
        return hljs.highlightAuto(code).value;
    },
    breaks: true,
});

const renderer = new marked.Renderer();
renderer.code = function (code, lang) {
    const text = typeof code === 'object' ? code.text : code;
    const language = typeof code === 'object' ? code.lang : lang;
    let highlighted;
    try {
        highlighted = language && hljs.getLanguage(language)
            ? hljs.highlight(text, { language }).value
            : hljs.highlightAuto(text).value;
    } catch {
        highlighted = text;
    }
    const langLabel = language || '';
    return `<div class="code-wrapper"><pre><code class="hljs language-${langLabel}">${highlighted}</code><button class="copy-btn" onclick="copyCode(this)">复制</button></pre></div>`;
};
marked.use({ renderer });

function init() {
    loadConversations();
    setupListeners();
}

function setupListeners() {
    newBtn.addEventListener('click', createNewConversation);

    sidebarToggle.addEventListener('click', () => {
        sidebarEl.classList.toggle('collapsed');
    });

    inputEl.addEventListener('input', () => {
        sendBtn.disabled = !inputEl.value.trim() || isStreaming;
        inputEl.style.height = 'auto';
        inputEl.style.height = Math.min(inputEl.scrollHeight, 150) + 'px';
    });

    inputEl.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            if (!sendBtn.disabled && !isStreaming) sendMessage();
        }
    });

    sendBtn.addEventListener('click', () => {
        if (!isStreaming && inputEl.value.trim()) sendMessage();
    });

    messagesEl.addEventListener('click', (e) => {
        if (e.target.classList.contains('copy-btn')) {
            copyCode(e.target);
        }
    });
}

async function loadConversations() {
    try {
        const res = await fetch(`${API_BASE}/api/conversations`);
        conversations = await res.json();
        renderConversationList();
    } catch (err) {
        console.error('Failed to load conversations:', err);
    }
}

function renderConversationList() {
    convListEl.innerHTML = '';
    for (const conv of conversations) {
        const item = document.createElement('div');
        item.className = 'conv-item' + (conv.id === activeConvId ? ' active' : '');
        item.dataset.id = conv.id;

        const title = document.createElement('span');
        title.className = 'conv-title';
        title.textContent = conv.title;

        const actions = document.createElement('div');
        actions.className = 'conv-actions';

        const rename = document.createElement('button');
        rename.className = 'conv-rename';
        rename.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"></path><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"></path></svg>`;
        rename.addEventListener('click', (e) => {
            e.stopPropagation();
            startRename(conv.id, title);
        });

        const del = document.createElement('button');
        del.className = 'conv-delete';
        del.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="3 6 5 6 21 6"></polyline><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path></svg>`;
        del.addEventListener('click', (e) => {
            e.stopPropagation();
            deleteConversation(conv.id);
        });

        actions.appendChild(rename);
        actions.appendChild(del);
        item.appendChild(title);
        item.appendChild(actions);
        item.addEventListener('click', () => switchConversation(conv.id));
        convListEl.appendChild(item);
    }
}

async function createNewConversation() {
    try {
        saveCurrentMessages();

        const res = await fetch(`${API_BASE}/api/conversations`, { method: 'POST' });
        const conv = await res.json();
        conversations.unshift(conv);
        activeConvId = conv.id;
        renderConversationList();
        clearChat();
        titleEl.textContent = conv.title;
        inputEl.focus();
    } catch (err) {
        console.error('Failed to create conversation:', err);
    }
}

async function switchConversation(convId) {
    if (convId === activeConvId || isStreaming) return;

    saveCurrentMessages();

    activeConvId = convId;
    renderConversationList();

    const conv = conversations.find(c => c.id === convId);
    if (conv) titleEl.textContent = conv.title;

    restoreMessages(convId);
}

async function deleteConversation(convId) {
    if (isStreaming) return;
    try {
        await fetch(`${API_BASE}/api/conversations/${convId}`, { method: 'DELETE' });
        conversations = conversations.filter(c => c.id !== convId);
        messageCache.delete(convId);

        if (activeConvId === convId) {
            activeConvId = conversations.length > 0 ? conversations[0].id : null;
            if (activeConvId) {
                const conv = conversations.find(c => c.id === activeConvId);
                if (conv) titleEl.textContent = conv.title;
                restoreMessages(activeConvId);
            } else {
                clearChat();
                titleEl.textContent = '方塘 AI';
            }
        }
        renderConversationList();
    } catch (err) {
        console.error('Failed to delete conversation:', err);
    }
}

function startRename(convId, titleEl) {
    const conv = conversations.find(c => c.id === convId);
    if (!conv || isStreaming) return;

    const input = document.createElement('input');
    input.type = 'text';
    input.className = 'conv-rename-input';
    input.value = conv.title;

    titleEl.replaceWith(input);
    input.focus();
    input.select();

    const commit = async () => {
        const newTitle = input.value.trim();
        if (!newTitle || newTitle === conv.title) {
            renderConversationList();
            return;
        }
        conv.title = newTitle;
        if (convId === activeConvId) {
            titleEl.textContent = newTitle;
            document.getElementById('current-title').textContent = newTitle;
        }
        renderConversationList();
        try {
            await fetch(`${API_BASE}/api/conversations/${convId}/title`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ title: newTitle }),
            });
        } catch { /* ignore */ }
    };

    input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') { e.preventDefault(); commit(); }
        if (e.key === 'Escape') { renderConversationList(); }
    });
    input.addEventListener('blur', commit);
}

function clearChat() {
    messagesEl.innerHTML = '';
    welcomeEl.style.display = '';
}

function saveCurrentMessages() {
    if (activeConvId && messagesEl.children.length > 0) {
        messageCache.set(activeConvId, messagesEl.innerHTML);
    }
}

function restoreMessages(convId) {
    const cached = messageCache.get(convId);
    if (cached) {
        messagesEl.innerHTML = cached;
        welcomeEl.style.display = 'none';
        scrollToBottom();
    } else {
        clearChat();
    }
}

function hideWelcome() {
    welcomeEl.style.display = 'none';
}

async function sendMessage() {
    const text = inputEl.value.trim();
    if (!text) return;

    if (!activeConvId) {
        await createNewConversation();
    }

    hideWelcome();
    inputEl.value = '';
    inputEl.style.height = 'auto';
    sendBtn.disabled = true;

    appendMessage('user', text);

    const conv = conversations.find(c => c.id === activeConvId);
    if (conv && conv.title === '新对话') {
        const newTitle = text.slice(0, 20) + (text.length > 20 ? '...' : '');
        conv.title = newTitle;
        titleEl.textContent = newTitle;
        renderConversationList();
        fetch(`${API_BASE}/api/conversations/${activeConvId}/title`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ title: newTitle }),
        }).catch(() => {});
    }

    await streamResponse(text);
}

async function streamResponse(message) {
    isStreaming = true;
    sendBtn.disabled = true;

    const assistantBubble = appendMessage('assistant', '');
    const contentEl = assistantBubble.querySelector('.message-content');
    let toolIndicator = null;

    const typing = document.createElement('div');
    typing.className = 'typing-indicator';
    typing.innerHTML = '<div class="dot"></div><div class="dot"></div><div class="dot"></div>';
    contentEl.appendChild(typing);
    scrollToBottom();

    try {
        const res = await fetch(`${API_BASE}/api/chat/${activeConvId}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message }),
        });

        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: 'Request failed' }));
            throw new Error(err.detail || 'Request failed');
        }

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let fullText = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop() || '';

            let eventType = '';
            for (const line of lines) {
                if (line.startsWith('event: ')) {
                    eventType = line.slice(7).trim();
                } else if (line.startsWith('data: ')) {
                    const dataStr = line.slice(6);
                    try {
                        const data = JSON.parse(dataStr);
                        handleSSE(eventType, data, contentEl, () => {
                            if (typing.parentNode) typing.remove();
                        }, () => {
                            toolIndicator = createToolIndicator(data.content);
                            contentEl.appendChild(toolIndicator);
                            scrollToBottom();
                        }, () => {
                            if (toolIndicator) {
                                toolIndicator.remove();
                                toolIndicator = null;
                            }
                        });

                        if (eventType === 'text' && data.content) {
                            fullText += data.content;
                        }
                    } catch {
                        /* skip malformed data */
                    }
                    eventType = '';
                }
            }
        }

        if (!fullText && !contentEl.textContent.trim()) {
            contentEl.innerHTML = renderMarkdown('（未生成文本输出）');
        } else if (fullText) {
            contentEl.innerHTML = renderMarkdown(fullText);
        }

    } catch (err) {
        if (typing.parentNode) typing.remove();
        contentEl.innerHTML = `<span style="color:#ef4444">Error: ${err.message}</span>`;
    } finally {
        isStreaming = false;
        sendBtn.disabled = !inputEl.value.trim();
        inputEl.focus();
        scrollToBottom();
    }
}

function handleSSE(event, data, container, onStart, onToolStart, onToolEnd) {
    switch (event) {
        case 'text':
            onStart();
            appendTextChunk(container, data.content);
            scrollToBottom();
            break;
        case 'tool':
            onStart();
            onToolStart();
            break;
        case 'tool_end':
            onToolEnd();
            break;
        case 'error':
            onStart();
            const errSpan = document.createElement('span');
            errSpan.style.color = '#ef4444';
            errSpan.textContent = data.content;
            container.appendChild(errSpan);
            break;
    }
}

function appendMessage(role, content) {
    const msg = document.createElement('div');
    msg.className = `message ${role}`;

    const bubble = document.createElement('div');
    bubble.className = 'message-bubble';

    const contentEl = document.createElement('div');
    contentEl.className = 'message-content';

    if (role === 'user') {
        contentEl.textContent = content;
    } else {
        contentEl.innerHTML = renderMarkdown(content);
    }

    bubble.appendChild(contentEl);
    msg.appendChild(bubble);
    messagesEl.appendChild(msg);
    scrollToBottom();

    return msg;
}

function appendTextChunk(container, text) {
    const currentHTML = container.innerHTML;
    const currentText = container.textContent || '';
    container.innerHTML = renderMarkdown(currentText + text);
}

function renderMarkdown(text) {
    if (!text) return '';
    return marked.parse(text);
}

function createToolIndicator(text) {
    const el = document.createElement('div');
    el.className = 'tool-indicator';
    el.innerHTML = `<div class="spinner"></div><span>${text}</span>`;
    return el;
}

function scrollToBottom() {
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

window.copyCode = function (btn) {
    const code = btn.parentElement.querySelector('code');
    const text = code.textContent;
    navigator.clipboard.writeText(text).then(() => {
        btn.textContent = '已复制';
        setTimeout(() => { btn.textContent = '复制'; }, 2000);
    }).catch(() => {
        btn.textContent = '失败';
        setTimeout(() => { btn.textContent = '复制'; }, 2000);
    });
};

init();
