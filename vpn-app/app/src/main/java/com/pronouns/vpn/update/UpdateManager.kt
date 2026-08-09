package com.pronouns.vpn.update

import com.pronouns.vpn.domain.model.UpdateManifest
import com.pronouns.vpn.domain.repository.UpdateRepository
import com.pronouns.vpn.security.MemoryZeroizer
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.withContext
import java.io.File
import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class UpdateManager @Inject constructor(
    private val updateRepository: UpdateRepository,
    private val memoryZeroizer: MemoryZeroizer
) {

    sealed interface UpdateState {
        data object Idle : UpdateState
        data object Checking : UpdateState
        data class Available(val manifest: UpdateManifest) : UpdateState
        data object Downloading : UpdateState
        data class DownloadProgress(val bytesDownloaded: Long, val totalBytes: Long) : UpdateState
        data object Verifying : UpdateState
        data object ReadyToInstall : UpdateState
        data class Error(val message: String) : UpdateState
    }

    private val _updateState = MutableStateFlow<UpdateState>(UpdateState.Idle)
    val updateState: StateFlow<UpdateState> = _updateState.asStateFlow()

    suspend fun checkForUpdate() {
        _updateState.value = UpdateState.Checking
        try {
            val manifest = updateRepository.checkForUpdate()
            if (manifest != null) {
                _updateState.value = UpdateState.Available(manifest)
            } else {
                _updateState.value = UpdateState.Idle
            }
        } catch (e: Exception) {
            _updateState.value = UpdateState.Error(
                "Update check failed: ${e.message}"
            )
        }
    }

    suspend fun downloadAndInstall(manifest: UpdateManifest) {
        _updateState.value = UpdateState.Downloading

        try {
            val apkFile = withContext(Dispatchers.IO) {
                updateRepository.downloadUpdate(manifest)
            }

            _updateState.value = UpdateState.Verifying

            val valid = withContext(Dispatchers.Default) {
                updateRepository.verifySignature(apkFile, manifest.signatureHash)
            }

            if (!valid) {
                apkFile.delete()
                _updateState.value = UpdateState.Error("Signature verification failed")
                return
            }

            _updateState.value = UpdateState.ReadyToInstall

            withContext(Dispatchers.Main) {
                updateRepository.installUpdate(apkFile)
            }
        } catch (e: Exception) {
            _updateState.value = UpdateState.Error(
                "Update failed: ${e.message}"
            )
        }
    }

    suspend fun checkAndApply() {
        checkForUpdate()
        val state = _updateState.value
        if (state is UpdateState.Available) {
            downloadAndInstall(state.manifest)
        }
    }

    fun reset() {
        _updateState.value = UpdateState.Idle
    }
}
