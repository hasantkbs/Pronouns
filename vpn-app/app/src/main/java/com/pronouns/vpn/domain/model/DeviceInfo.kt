package com.pronouns.vpn.domain.model

data class DeviceInfo(
    val deviceId: String,
    val androidId: String,
    val manufacturer: String,
    val model: String,
    val androidVersion: String,
    val appVersion: String,
    val installTimestamp: Long,
    val appSignatureHash: String
)
