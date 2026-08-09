package com.pronouns.vpn.vpn

import com.pronouns.vpn.domain.model.VpnCredentials
import com.pronouns.vpn.domain.model.VpnProfile
import kotlinx.coroutines.flow.Flow

enum class VpnState {
    DISCONNECTED,
    CONNECTING,
    CONNECTED,
    DISCONNECTING,
    ERROR
}

interface OpenVpnManager {
    suspend fun connect(profile: VpnProfile, credentials: VpnCredentials): Result<Unit>
    suspend fun disconnect()
    fun getConnectionStatus(): Flow<VpnState>
    suspend fun isConnected(): Boolean
    fun getLastError(): Flow<String?>
}
