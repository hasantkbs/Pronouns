package com.pronouns.vpn.presentation.ui.components

import androidx.compose.animation.animateColorAsState
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.pronouns.vpn.domain.model.VpnStatus

@Composable
fun VpnStatusCard(
    status: VpnStatus,
    modifier: Modifier = Modifier
) {
    val statusColor by animateColorAsState(
        targetValue = when (status) {
            is VpnStatus.Connected -> Color(0xFF4CAF50)
            is VpnStatus.Error -> MaterialTheme.colorScheme.error
            is VpnStatus.Disconnected -> MaterialTheme.colorScheme.onSurfaceVariant
            else -> MaterialTheme.colorScheme.primary
        },
        label = "statusColor"
    )

    val statusText = when (status) {
        is VpnStatus.Disconnected -> "Disconnected"
        is VpnStatus.Connecting -> "Connecting..."
        is VpnStatus.Authenticating -> "Authenticating..."
        is VpnStatus.AcquiringIp -> "Acquiring IP..."
        is VpnStatus.Connected -> "Connected"
        is VpnStatus.Error -> "Error"
        is VpnStatus.Reconnecting -> "Reconnecting..."
        is VpnStatus.Disconnecting -> "Disconnecting..."
    }

    val statusIcon = when (status) {
        is VpnStatus.Connected -> "●"
        is VpnStatus.Error -> "✕"
        is VpnStatus.Disconnected -> "○"
        else -> "◌"
    }

    Card(
        modifier = modifier,
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surface
        ),
        elevation = CardDefaults.cardElevation(defaultElevation = 2.dp)
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(20.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            Column {
                Text(
                    text = "VPN Status",
                    style = MaterialTheme.typography.titleSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
                Spacer(modifier = Modifier.height(4.dp))
                Text(
                    text = statusText,
                    style = MaterialTheme.typography.titleLarge,
                    fontWeight = FontWeight.Bold,
                    color = statusColor
                )
            }

            when (status) {
                is VpnStatus.Connecting,
                is VpnStatus.Authenticating,
                is VpnStatus.AcquiringIp,
                is VpnStatus.Reconnecting -> {
                    CircularProgressIndicator(
                        modifier = Modifier.size(32.dp),
                        color = statusColor,
                        strokeWidth = 3.dp
                    )
                }
                is VpnStatus.Connected -> {
                    Text(
                        text = statusIcon,
                        fontSize = MaterialTheme.typography.headlineLarge.fontSize,
                        color = statusColor
                    )
                }
                else -> {
                    Text(
                        text = statusIcon,
                        fontSize = MaterialTheme.typography.headlineLarge.fontSize,
                        color = statusColor
                    )
                }
            }
        }

        if (status is VpnStatus.Connected) {
            Text(
                text = "Local IP: ${status.localIp}",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.padding(start = 20.dp, bottom = 12.dp)
            )
        }

        if (status is VpnStatus.Error) {
            Text(
                text = status.message,
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.error,
                modifier = Modifier.padding(start = 20.dp, bottom = 12.dp, end = 20.dp)
            )
        }
    }
}
