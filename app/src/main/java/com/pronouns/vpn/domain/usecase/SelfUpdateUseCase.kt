package com.pronouns.vpn.domain.usecase

import android.content.Context
import android.content.pm.PackageManager
import android.os.Build
import com.pronouns.vpn.data.repository.UpdateRepository
import com.pronouns.vpn.domain.model.UpdateManifest
import dagger.hilt.android.qualifiers.ApplicationContext
import javax.inject.Inject

sealed class UpdateResult {
    object NoUpdate : UpdateResult()
    data class UpdateAvailable(val manifest: UpdateManifest) : UpdateResult()
    object UpdateInstalled : UpdateResult()
    data class Error(val message: String) : UpdateResult()
}

class SelfUpdateUseCase @Inject constructor(
    private val updateRepository: UpdateRepository,
    @ApplicationContext private val context: Context
) {
    suspend operator fun invoke(): Result<UpdateResult> {
        return runCatching {
            val manifest = updateRepository.checkForUpdates().getOrThrow()

            if (manifest.versionCode <= currentVersionCode()) {
                return@runCatching UpdateResult.NoUpdate
            }

            val file = updateRepository.downloadUpdate(manifest).getOrThrow()

            val hashToVerify = if (manifest.deltaUrl != null && manifest.deltaSignatureHash != null) {
                manifest.deltaSignatureHash
            } else {
                manifest.signatureHash
            }

            val isValid = updateRepository.verifyUpdateSignature(file, hashToVerify)
            if (!isValid) {
                file.delete()
                return@runCatching UpdateResult.Error("Update signature verification failed")
            }

            updateRepository.installUpdate(file).getOrThrow()
            UpdateResult.UpdateInstalled
        }
    }

    private fun currentVersionCode(): Int {
        return try {
            val info = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                context.packageManager.getPackageInfo(
                    context.packageName,
                    PackageManager.PackageInfoFlags.of(0L)
                )
            } else {
                @Suppress("DEPRECATION")
                context.packageManager.getPackageInfo(context.packageName, 0)
            }
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
                info.longVersionCode.toInt()
            } else {
                @Suppress("DEPRECATION")
                info.versionCode
            }
        } catch (_: PackageManager.NameNotFoundException) {
            0
        }
    }
}
