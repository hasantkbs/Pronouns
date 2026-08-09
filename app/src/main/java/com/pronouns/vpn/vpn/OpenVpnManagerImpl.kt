package com.pronouns.vpn.vpn

import android.content.Context
import android.content.Intent
import com.pronouns.vpn.domain.model.VpnCredentials
import com.pronouns.vpn.domain.model.VpnProfile
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asSharedFlow
import kotlinx.coroutines.flow.asStateFlow
import net.openvpn.v3.openvpn.OpenVpnApi
import net.openvpn.v3.openvpn.OpenVPNStatus
import net.openvpn.v3.openvpn.OpenVPNStatusListener
import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class OpenVpnManagerImpl @Inject constructor(
    @ApplicationContext private val context: Context
) : OpenVpnManager {

    private val _connectionStatus = MutableStateFlow(VpnState.DISCONNECTED)
    private val _lastError = MutableSharedFlow<String?>(replay = 0, extraBufferCapacity = 1)

    private var retryCount = 0

    private val statusListener = OpenVPNStatusListener { status ->
        handleStatusUpdate(status)
    }

    init {
        OpenVpnApi.addStatusListener(statusListener)
    }

    override suspend fun connect(profile: VpnProfile, credentials: VpnCredentials): Result<Unit> {
        return runCatching {
            if (_connectionStatus.value == VpnState.CONNECTED ||
                _connectionStatus.value == VpnState.CONNECTING
            ) {
                disconnectInternal()
                delay(500)
            }

            retryCount = 0
            attemptConnection(profile, credentials)
        }
    }

    private suspend fun attemptConnection(
        profile: VpnProfile,
        credentials: VpnCredentials,
        isRetry: Boolean = false
    ) {
        _connectionStatus.value = VpnState.CONNECTING

        val configText = buildConfigString(profile)
        val credsString = "${credentials.username}\n${credentials.password}"

        val intent = Intent(context, PronounsVpnService::class.java).apply {
            putExtra(CONFIG_EXTRA, configText)
            putExtra(CREDENTIALS_EXTRA, credsString)
        }

        try {
            OpenVpnApi.startVpn(context, configText, credsString, intent)
        } catch (e: Exception) {
            if (!isRetry && retryCount < MAX_RETRIES) {
                retryCount++
                delay(RETRY_DELAY_MS * retryCount)
                attemptConnection(profile, credentials, isRetry = true)
            } else {
                _connectionStatus.value = VpnState.ERROR
                _lastError.tryEmit(e.message ?: "Connection failed")
                throw e
            }
        }
    }

    override suspend fun disconnect() {
        disconnectInternal()
    }

    private suspend fun disconnectInternal() {
        _connectionStatus.value = VpnState.DISCONNECTING
        try {
            OpenVpnApi.stopVpn(context)
        } catch (_: Exception) {
        }
        _connectionStatus.value = VpnState.DISCONNECTED
    }

    override fun getConnectionStatus(): Flow<VpnState> = _connectionStatus.asStateFlow()

    override suspend fun isConnected(): Boolean =
        _connectionStatus.value == VpnState.CONNECTED

    override fun getLastError(): Flow<String?> = _lastError.asSharedFlow()

    private fun handleStatusUpdate(status: OpenVPNStatus) {
        val state = mapState(status.state)
        _connectionStatus.value = state

        if (state == VpnState.ERROR) {
            _lastError.tryEmit(status.message ?: status.state)
        }
    }

    private fun mapState(openVpnState: String): VpnState {
        return when (openVpnState) {
            "CONNECTED" -> VpnState.CONNECTED
            "CONNECTING" -> VpnState.CONNECTING
            "DISCONNECTED" -> VpnState.DISCONNECTED
            "DISCONNECTING" -> VpnState.DISCONNECTING
            "WAIT" -> VpnState.CONNECTING
            "RECONNECTING" -> VpnState.CONNECTING
            "EXITING" -> VpnState.DISCONNECTING
            "RESOLVE" -> VpnState.CONNECTING
            "TCP_CONNECT" -> VpnState.CONNECTING
            "AUTH" -> VpnState.CONNECTING
            "GET_CONFIG" -> VpnState.CONNECTING
            "ASSIGN_IP" -> VpnState.CONNECTING
            "ADD_ROUTES" -> VpnState.CONNECTING
            else -> {
                val errorStates = listOf(
                    "AUTH_FAILED", "TLS_ERROR", "CERT_VERIFY_FAIL",
                    "CONNECTION_TIMEOUT", "PROXY_ERROR", "KEEPALIVE_TIMEOUT"
                )
                if (errorStates.any { openVpnState.startsWith(it) }) {
                    VpnState.ERROR
                } else {
                    VpnState.ERROR
                }
            }
        }
    }

    private fun buildConfigString(profile: VpnProfile): String {
        return buildString {
            appendLine("client")
            appendLine("dev tun")
            appendLine("proto ${profile.protocol}")
            appendLine("remote ${profile.serverHost} ${profile.serverPort}")
            appendLine("resolv-retry infinite")
            appendLine("nobind")
            appendLine("persist-key")
            appendLine("persist-tun")
            appendLine("remote-cert-tls server")
            appendLine("auth SHA256")
            appendLine("verb 3")

            profile.cipher?.let { appendLine("cipher $it") }
            profile.caCert?.let {
                appendLine("<ca>")
                appendLine(it.trim())
                appendLine("</ca>")
            }
            profile.clientCert?.let {
                appendLine("<cert>")
                appendLine(it.trim())
                appendLine("</cert>")
            }
            profile.clientKey?.let {
                appendLine("<key>")
                appendLine(it.trim())
                appendLine("</key>")
            }
            profile.tlsAuth?.let {
                appendLine("<tls-auth>")
                appendLine(it.trim())
                appendLine("</tls-auth>")
                appendLine("key-direction 1")
            }
            for ((key, value) in profile.extraOptions) {
                appendLine("$key $value")
            }
            appendLine("auth-user-pass")
        }
    }

    private companion object {
        private const val CONFIG_EXTRA = "config"
        private const val CREDENTIALS_EXTRA = "credentials"
        private const val MAX_RETRIES = 3
        private const val RETRY_DELAY_MS = 1000L
    }
}
