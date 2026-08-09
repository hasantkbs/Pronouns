package com.pronouns.vpn.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle

@Composable
fun MainScreen(
    viewModel: MainViewModel = hiltViewModel()
) {
    val connectionState by viewModel.connectionState.collectAsStateWithLifecycle()
    val errorState by viewModel.errorState.collectAsStateWithLifecycle()
    val updateAvailable by viewModel.updateAvailable.collectAsStateWithLifecycle()

    Scaffold(
        topBar = {
            if (updateAvailable) {
                Surface(
                    color = MaterialTheme.colorScheme.tertiaryContainer,
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Row(
                        modifier = Modifier.padding(horizontal = 16.dp, vertical = 12.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Text(
                            text = "Update available",
                            modifier = Modifier.weight(1f),
                            style = MaterialTheme.typography.bodyMedium
                        )
                        TextButton(onClick = { viewModel.installUpdate() }) {
                            Text("Install")
                        }
                    }
                }
            }
        }
    ) { padding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Center
        ) {
            val statusText = when (connectionState) {
                VpnConnectionState.DISCONNECTED -> "Disconnected"
                VpnConnectionState.CONNECTING -> "Connecting"
                VpnConnectionState.CONNECTED -> "Connected"
            }

            Text(
                text = statusText,
                fontSize = 28.sp,
                fontWeight = FontWeight.Bold,
                color = when (connectionState) {
                    VpnConnectionState.CONNECTED -> MaterialTheme.colorScheme.primary
                    VpnConnectionState.CONNECTING -> MaterialTheme.colorScheme.tertiary
                    VpnConnectionState.DISCONNECTED -> MaterialTheme.colorScheme.onSurfaceVariant
                }
            )

            Spacer(modifier = Modifier.height(32.dp))

            Button(
                onClick = {
                    when (connectionState) {
                        VpnConnectionState.DISCONNECTED -> viewModel.onConnect()
                        VpnConnectionState.CONNECTED -> viewModel.onDisconnect()
                        VpnConnectionState.CONNECTING -> {}
                    }
                },
                enabled = connectionState != VpnConnectionState.CONNECTING
            ) {
                Text(
                    text = when (connectionState) {
                        VpnConnectionState.DISCONNECTED -> "Connect"
                        VpnConnectionState.CONNECTING -> "Connecting..."
                        VpnConnectionState.CONNECTED -> "Disconnect"
                    },
                    fontSize = 18.sp
                )
            }

            if (errorState != null) {
                Spacer(modifier = Modifier.height(16.dp))
                Text(
                    text = errorState!!,
                    color = MaterialTheme.colorScheme.error,
                    style = MaterialTheme.typography.bodyMedium
                )
            }
        }
    }
}
