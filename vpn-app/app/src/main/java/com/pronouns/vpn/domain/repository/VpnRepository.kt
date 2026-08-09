package com.pronouns.vpn.domain.repository

import com.pronouns.vpn.domain.model.VpnProfile
import com.pronouns.vpn.domain.model.VpnStatus
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.StateFlow

interface VpnRepository {
    val vpnStatus: StateFlow<VpnStatus>
    val trafficStats: Flow<Pair<Long, Long>>

    suspend fun loadProfile(): VpnProfile?
    suspend fun connect(credentials: String, password: String)
    suspend fun disconnect()
    suspend fun reconnect()
    fun isConnected(): Boolean
    fun isConnecting(): Boolean
    suspend fun verifyTunnel(): Boolean
    suspend fun getConnectionDurationMs(): Long
}
