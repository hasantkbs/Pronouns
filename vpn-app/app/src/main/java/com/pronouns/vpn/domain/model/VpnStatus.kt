package com.pronouns.vpn.domain.model

sealed interface VpnStatus {
    data object Disconnected : VpnStatus
    data object Connecting : VpnStatus
    data object Authenticating : VpnStatus
    data object AcquiringIp : VpnStatus
    data class Connected(
        val localIp: String,
        val remoteIp: String,
        val bytesIn: Long = 0,
        val bytesOut: Long = 0,
        val durationMs: Long = 0
    ) : VpnStatus

    data class Error(
        val throwable: Throwable,
        val message: String
    ) : VpnStatus

    data object Reconnecting : VpnStatus
    data object Disconnecting : VpnStatus
}
