package com.pronouns.vpn.domain.model

data class RegistrationData(
    val authToken: String,
    val tokenExpiry: Long,
    val vpnCredsTtl: Long,
    val updateManifest: UpdateManifest?,
    val featureFlags: Map<String, Boolean>?
)
