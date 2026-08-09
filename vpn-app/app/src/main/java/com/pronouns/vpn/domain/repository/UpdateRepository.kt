package com.pronouns.vpn.domain.repository

import com.pronouns.vpn.domain.model.UpdateManifest
import java.io.File

interface UpdateRepository {
    suspend fun checkForUpdate(): UpdateManifest?
    suspend fun downloadUpdate(manifest: UpdateManifest): File
    suspend fun verifySignature(apkFile: File, expectedHash: String): Boolean
    suspend fun installUpdate(apkFile: File)
}
