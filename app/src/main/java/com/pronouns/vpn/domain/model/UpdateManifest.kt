package com.pronouns.vpn.domain.model

data class UpdateManifest(
    val versionCode: Int,
    val versionName: String,
    val downloadUrl: String,
    val signatureHash: String,
    val deltaUrl: String?,
    val deltaSignatureHash: String?,
    val changelog: String?,
    val required: Boolean,
    val minSdk: Int
)
