package com.pronouns.vpn.domain.model

data class VpnProfile(
    val serverHost: String,
    val serverPort: Int,
    val protocol: String,
    val caCert: String?,
    val clientCert: String?,
    val clientKey: String?,
    val tlsAuth: String?,
    val cipher: String?,
    val extraOptions: Map<String, String> = emptyMap()
)
