package com.fangtang.ai.ui

import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.net.Uri
import android.os.Build
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.Edit
import androidx.compose.material.icons.filled.Image
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Slider
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import com.fangtang.ai.api.ApiClient
import com.fangtang.ai.model.Conversation
import com.fangtang.ai.viewmodel.ChatViewModel

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ConversationListDrawer(
    viewModel: ChatViewModel,
    onClose: () -> Unit,
) {
    val showSettingsState = remember { mutableStateOf(false) }
    val renamingConvId = remember { mutableStateOf<String?>(null) }

    Column(modifier = Modifier.fillMaxSize()) {
        TopAppBar(
            title = {
                Text("方塘", fontWeight = FontWeight.Bold)
            },
            actions = {
                IconButton(onClick = { showSettingsState.value = true }) {
                    Icon(Icons.Default.Settings, contentDescription = "设置")
                }
                IconButton(onClick = onClose) {
                    Icon(Icons.Default.Close, contentDescription = "关闭")
                }
            },
            colors = TopAppBarDefaults.topAppBarColors(
                containerColor = MaterialTheme.colorScheme.surface,
            ),
        )

        LazyColumn(modifier = Modifier.fillMaxSize()) {
            item {
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .clickable {
                            viewModel.createAndSelectConversation()
                            onClose()
                        }
                        .padding(16.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Icon(
                        Icons.Default.Add,
                        contentDescription = null,
                        tint = MaterialTheme.colorScheme.primary,
                    )
                    Spacer(Modifier.width(16.dp))
                    Text(
                        text = "新对话",
                        style = MaterialTheme.typography.bodyLarge,
                        color = MaterialTheme.colorScheme.primary,
                        fontWeight = FontWeight.Medium,
                    )
                }
                HorizontalDivider(color = MaterialTheme.colorScheme.surfaceVariant)
            }

            items(viewModel.conversations) { conv ->
                ConversationDrawerItem(
                    conversation = conv,
                    isSelected = conv.id == viewModel.currentConvId,
                    onClick = {
                        viewModel.selectConversation(conv.id)
                        onClose()
                    },
                    onDelete = { viewModel.deleteConversation(conv.id) },
                    onRename = { renamingConvId.value = conv.id },
                )
                HorizontalDivider(color = MaterialTheme.colorScheme.surfaceVariant)
            }
        }
    }

    if (showSettingsState.value) {
        SettingsDialog(
            currentUrl = ApiClient.baseUrl,
            currentFontSize = viewModel.fontSizeScale,
            isDarkTheme = viewModel.isDarkTheme,
            backgroundImage = viewModel.backgroundImage,
            onDismiss = { showSettingsState.value = false },
            onSave = { url, fontSize, darkTheme, background ->
                viewModel.updateServerUrl(url)
                viewModel.updateFontSize(fontSize)
                if (viewModel.isDarkTheme != darkTheme) viewModel.toggleTheme()
                viewModel.updateBackgroundImage(background)
                showSettingsState.value = false
            },
        )
    }

    if (renamingConvId.value != null) {
        val conv = viewModel.conversations.find { it.id == renamingConvId.value }
        if (conv != null) {
            RenameDialog(
                currentTitle = conv.title,
                onDismiss = { renamingConvId.value = null },
                onConfirm = { newTitle ->
                    viewModel.renameConversation(conv.id, newTitle)
                    renamingConvId.value = null
                },
            )
        }
    }

    if (viewModel.errorMessage != null) {
        AlertDialog(
            onDismissRequest = { viewModel.clearError() },
            title = { Text("错误") },
            text = { Text(viewModel.errorMessage ?: "未知错误") },
            confirmButton = {
                TextButton(onClick = { viewModel.clearError() }) {
                    Text("确定")
                }
            },
        )
    }
}

@Composable
private fun ConversationDrawerItem(
    conversation: Conversation,
    isSelected: Boolean,
    onClick: () -> Unit,
    onDelete: () -> Unit,
    onRename: () -> Unit,
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clickable(onClick = onClick)
            .background(
                if (isSelected) MaterialTheme.colorScheme.primaryContainer.copy(alpha = 0.3f)
                else MaterialTheme.colorScheme.surface
            )
            .padding(horizontal = 16.dp, vertical = 12.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Column(modifier = Modifier.weight(1f)) {
            Text(
                text = conversation.title,
                style = MaterialTheme.typography.bodyLarge,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis,
            )
            Spacer(Modifier.height(4.dp))
            Text(
                text = conversation.created_at,
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.7f),
            )
        }
        IconButton(onClick = onRename) {
            Icon(
                Icons.Default.Edit,
                contentDescription = "重命名",
                modifier = Modifier.size(20.dp),
                tint = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
        IconButton(onClick = onDelete) {
            Icon(
                Icons.Default.Delete,
                contentDescription = "删除",
                modifier = Modifier.size(20.dp),
                tint = MaterialTheme.colorScheme.error,
            )
        }
    }
}

@Composable
private fun RenameDialog(
    currentTitle: String,
    onDismiss: () -> Unit,
    onConfirm: (String) -> Unit,
) {
    val titleState = remember { mutableStateOf(currentTitle) }

    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("重命名对话") },
        text = {
            OutlinedTextField(
                value = titleState.value,
                onValueChange = { titleState.value = it },
                modifier = Modifier.fillMaxWidth(),
                singleLine = true,
                placeholder = { Text("输入新名称") },
            )
        },
        confirmButton = {
            TextButton(onClick = { onConfirm(titleState.value) }) {
                Text("确定")
            }
        },
        dismissButton = {
            TextButton(onClick = onDismiss) {
                Text("取消")
            }
        },
    )
}

@Composable
private fun SettingsDialog(
    currentUrl: String,
    currentFontSize: Float,
    isDarkTheme: Boolean,
    backgroundImage: Bitmap?,
    onDismiss: () -> Unit,
    onSave: (String, Float, Boolean, Bitmap?) -> Unit,
) {
    val urlState = remember { mutableStateOf(currentUrl) }
    val fontSizeState = remember { mutableStateOf(currentFontSize) }
    val darkThemeState = remember { mutableStateOf(isDarkTheme) }
    val backgroundState = remember { mutableStateOf(backgroundImage) }
    val context = LocalContext.current

    val imagePickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        uri?.let {
            val bitmap = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
                val source = ImageDecoder.createSource(context.contentResolver, uri)
                ImageDecoder.decodeBitmap(source)
            } else {
                context.contentResolver.openInputStream(uri)?.use { inputStream ->
                    android.graphics.BitmapFactory.decodeStream(inputStream)
                }
            }
            backgroundState.value = bitmap
        }
    }

    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("设置") },
        text = {
            Column {
                Text(
                    text = "后端 API 地址",
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
                Spacer(Modifier.height(8.dp))
                OutlinedTextField(
                    value = urlState.value,
                    onValueChange = { urlState.value = it },
                    modifier = Modifier.fillMaxWidth(),
                    singleLine = true,
                    placeholder = { Text("http://192.168.1.100:8000") },
                )
                Spacer(Modifier.height(8.dp))
                Text(
                    text = "模拟器用 10.0.2.2，真机用电脑局域网 IP",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.7f),
                )
                Spacer(Modifier.height(16.dp))

                Text(
                    text = "字体大小: ${(fontSizeState.value * 100).toInt()}%",
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
                Spacer(Modifier.height(8.dp))
                Slider(
                    value = fontSizeState.value,
                    onValueChange = { fontSizeState.value = it },
                    valueRange = 0.8f..1.5f,
                    steps = 6,
                )
                Spacer(Modifier.height(16.dp))

                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Text(
                        text = "深色主题",
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                    Switch(
                        checked = darkThemeState.value,
                        onCheckedChange = { darkThemeState.value = it },
                    )
                }
                Spacer(Modifier.height(16.dp))

                Text(
                    text = "聊天背景",
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
                Spacer(Modifier.height(8.dp))
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    TextButton(
                        onClick = { imagePickerLauncher.launch("image/*") },
                        modifier = Modifier.weight(1f),
                    ) {
                        Icon(Icons.Default.Image, contentDescription = null, modifier = Modifier.size(18.dp))
                        Spacer(Modifier.width(4.dp))
                        Text(if (backgroundState.value != null) "更换背景" else "选择背景")
                    }
                    if (backgroundState.value != null) {
                        TextButton(
                            onClick = { backgroundState.value = null },
                            modifier = Modifier.weight(1f),
                        ) {
                            Icon(Icons.Default.Delete, contentDescription = null, modifier = Modifier.size(18.dp))
                            Spacer(Modifier.width(4.dp))
                            Text("移除背景")
                        }
                    }
                }
            }
        },
        confirmButton = {
            TextButton(onClick = { onSave(urlState.value, fontSizeState.value, darkThemeState.value, backgroundState.value) }) {
                Text("保存")
            }
        },
        dismissButton = {
            TextButton(onClick = onDismiss) {
                Text("取消")
            }
        },
    )
}
