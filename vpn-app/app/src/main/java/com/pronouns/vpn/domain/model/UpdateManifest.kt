package com.pronouns.vpn.domain.model

data class UpdateManifest(
    val latestVersionCode: Int,
    val latestVersionName: String,
    val minVersionCode: Int,
    val downloadUrl: String,
    val downloadUrlFallback: String?,
    val deltaUrl: String?,
    val signatureHash: String,
    val fileSizeBytes: Long,
    val releaseNotes: String,
    val isCritical: Boolean,
    val rolloutPercentage: Int
) {
    fun isAvailable(currentVersionCode: Int): Boolean =
        latestVersionCode > currentVersionCode
}
