package com.fangtang.ai.ui

import android.os.Build
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Typography
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.dynamicDarkColorScheme
import androidx.compose.material3.dynamicLightColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.unit.sp

private val LightColors = lightColorScheme(
    primary = Color(0xFF1B6EF0),
    onPrimary = Color.White,
    primaryContainer = Color(0xFFD8E2FF),
    secondary = Color(0xFF5A5F72),
    background = Color(0xFFF8F9FE),
    surface = Color.White,
    surfaceVariant = Color(0xFFF0F1F5),
)

private val DarkColors = darkColorScheme(
    primary = Color(0xFFADC6FF),
    onPrimary = Color(0xFF002E6E),
    primaryContainer = Color(0xFF00449C),
    secondary = Color(0xFFC3C6DD),
    background = Color(0xFF1A1C1E),
    surface = Color(0xFF2B2D30),
    surfaceVariant = Color(0xFF3A3C3F),
)

@Composable
fun FangTangTheme(
    darkTheme: Boolean = isSystemInDarkTheme(),
    fontSizeScale: Float = 1.0f,
    content: @Composable () -> Unit,
) {
    val colors = when {
        Build.VERSION.SDK_INT >= Build.VERSION_CODES.S -> {
            val ctx = LocalContext.current
            if (darkTheme) dynamicDarkColorScheme(ctx) else dynamicLightColorScheme(ctx)
        }
        darkTheme -> DarkColors
        else -> LightColors
    }

    val defaultTypography = MaterialTheme.typography
    val scaledTypography = Typography(
        bodySmall = defaultTypography.bodySmall.copy(fontSize = defaultTypography.bodySmall.fontSize * fontSizeScale),
        bodyMedium = defaultTypography.bodyMedium.copy(fontSize = defaultTypography.bodyMedium.fontSize * fontSizeScale),
        bodyLarge = defaultTypography.bodyLarge.copy(fontSize = defaultTypography.bodyLarge.fontSize * fontSizeScale),
        titleSmall = defaultTypography.titleSmall.copy(fontSize = defaultTypography.titleSmall.fontSize * fontSizeScale),
        titleMedium = defaultTypography.titleMedium.copy(fontSize = defaultTypography.titleMedium.fontSize * fontSizeScale),
        titleLarge = defaultTypography.titleLarge.copy(fontSize = defaultTypography.titleLarge.fontSize * fontSizeScale),
    )

    MaterialTheme(
        colorScheme = colors,
        typography = scaledTypography,
        content = content,
    )
}
