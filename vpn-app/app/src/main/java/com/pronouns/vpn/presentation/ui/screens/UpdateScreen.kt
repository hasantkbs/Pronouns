package com.pronouns.vpn.presentation.ui.screens

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.Download
import androidx.compose.material.icons.filled.Error
import androidx.compose.material.icons.filled.SystemUpdate
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import com.pronouns.vpn.update.UpdateManager
import javax.inject.Inject

@Composable
fun UpdateScreen(
    updateManager: UpdateManager
) {
    val updateState by updateManager.updateState.collectAsState()

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp)
            .verticalScroll(rememberScrollState()),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text(
            text = "Update Manager",
            style = MaterialTheme.typography.headlineSmall,
            fontWeight = FontWeight.Bold
        )

        Spacer(modifier = Modifier.height(24.dp))

        when (val state = updateState) {
            is UpdateManager.UpdateState.Idle -> {
                IdleContent(onCheck = { /* check for update */ })
            }
            is UpdateManager.UpdateState.Checking -> {
                CheckingContent()
            }
            is UpdateManager.UpdateState.Available -> {
                AvailableContent(
                    versionName = state.manifest.latestVersionName,
                    releaseNotes = state.manifest.releaseNotes,
                    isCritical = state.manifest.isCritical,
                    onDownload = { updateManager.downloadAndInstall(state.manifest) }
                )
            }
            is UpdateManager.UpdateState.Downloading -> {
                DownloadingContent()
            }
            is UpdateManager.UpdateState.Verifying -> {
                VerifyingContent()
            }
            is UpdateManager.UpdateState.ReadyToInstall -> {
                ReadyToInstallContent(
                    onInstall = { updateManager.checkAndApply() }
                )
            }
            is UpdateManager.UpdateState.Error -> {
                ErrorContent(
                    message = state.message,
                    onRetry = { updateManager.checkForUpdate() }
                )
            }
            is UpdateManager.UpdateState.DownloadProgress -> {
                DownloadProgressContent(
                    bytesDownloaded = state.bytesDownloaded,
                    totalBytes = state.totalBytes
                )
            }
        }
    }
}

@Composable
private fun IdleContent(onCheck: () -> Unit) {
    Column(
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Icon(
            imageVector = Icons.Default.CheckCircle,
            contentDescription = null,
            modifier = Modifier.size(64.dp),
            tint = MaterialTheme.colorScheme.primary
        )
        Spacer(modifier = Modifier.height(16.dp))
        Text("App is up to date")
        Spacer(modifier = Modifier.height(24.dp))
        Button(onClick = onCheck) {
            Icon(Icons.Default.SystemUpdate, contentDescription = null)
            Spacer(modifier = Modifier.width(8.dp))
            Text("Check for Updates")
        }
    }
}

@Composable
private fun CheckingContent() {
    Column(
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        CircularProgressIndicator()
        Spacer(modifier = Modifier.height(16.dp))
        Text("Checking for updates...")
    }
}

@Composable
private fun AvailableContent(
    versionName: String,
    releaseNotes: String,
    isCritical: Boolean,
    onDownload: () -> Unit
) {
    Card(
        colors = CardDefaults.cardColors(
            containerColor = if (isCritical)
                MaterialTheme.colorScheme.errorContainer
            else
                MaterialTheme.colorScheme.primaryContainer
        )
    ) {
        Column(
            modifier = Modifier.padding(16.dp)
        ) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                if (isCritical) {
                    Icon(
                        Icons.Default.Error,
                        contentDescription = null,
                        tint = MaterialTheme.colorScheme.error
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                }
                Text(
                    text = "Update Available",
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.Bold
                )
            }
            Spacer(modifier = Modifier.height(8.dp))
            Text("Version: $versionName")
            if (isCritical) {
                Text(
                    text = "Critical security update",
                    color = MaterialTheme.colorScheme.error,
                    fontWeight = FontWeight.Bold
                )
            }
            Spacer(modifier = Modifier.height(8.dp))
            Text(
                text = releaseNotes,
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
            Spacer(modifier = Modifier.height(16.dp))
            Button(
                onClick = onDownload,
                modifier = Modifier.fillMaxWidth()
            ) {
                Icon(Icons.Default.Download, contentDescription = null)
                Spacer(modifier = Modifier.width(8.dp))
                Text("Download & Install")
            }
        }
    }
}

@Composable
private fun DownloadingContent() {
    Column(horizontalAlignment = Alignment.CenterHorizontally) {
        CircularProgressIndicator()
        Spacer(modifier = Modifier.height(16.dp))
        Text("Downloading update...")
    }
}

@Composable
private fun DownloadProgressContent(
    bytesDownloaded: Long,
    totalBytes: Long
) {
    val progress = if (totalBytes > 0) bytesDownloaded.toFloat() / totalBytes else 0f
    Column(
        modifier = Modifier.fillMaxWidth(),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        LinearProgressIndicator(
            progress = { progress },
            modifier = Modifier.fillMaxWidth()
        )
        Spacer(modifier = Modifier.height(8.dp))
        Text(
            text = "${bytesDownloaded / 1024} KB / ${totalBytes / 1024} KB",
            style = MaterialTheme.typography.bodySmall
        )
    }
}

@Composable
private fun VerifyingContent() {
    Column(horizontalAlignment = Alignment.CenterHorizontally) {
        CircularProgressIndicator()
        Spacer(modifier = Modifier.height(16.dp))
        Text("Verifying package signature...")
    }
}

@Composable
private fun ReadyToInstallContent(onInstall: () -> Unit) {
    Column(horizontalAlignment = Alignment.CenterHorizontally) {
        Icon(
            imageVector = Icons.Default.CheckCircle,
            contentDescription = null,
            modifier = Modifier.size(64.dp),
            tint = MaterialTheme.colorScheme.primary
        )
        Spacer(modifier = Modifier.height(16.dp))
        Text("Update downloaded and verified")
        Spacer(modifier = Modifier.height(24.dp))
        Button(onClick = onInstall) {
            Text("Install Now")
        }
    }
}

@Composable
private fun ErrorContent(
    message: String,
    onRetry: () -> Unit
) {
    Column(horizontalAlignment = Alignment.CenterHorizontally) {
        Icon(
            imageVector = Icons.Default.Error,
            contentDescription = null,
            modifier = Modifier.size(64.dp),
            tint = MaterialTheme.colorScheme.error
        )
        Spacer(modifier = Modifier.height(16.dp))
        Text(
            text = message,
            color = MaterialTheme.colorScheme.error,
            style = MaterialTheme.typography.bodyMedium
        )
        Spacer(modifier = Modifier.height(24.dp))
        TextButton(onClick = onRetry) {
            Text("Retry")
        }
    }
}
