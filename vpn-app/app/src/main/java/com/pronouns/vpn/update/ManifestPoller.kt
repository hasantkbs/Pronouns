package com.pronouns.vpn.update

import com.pronouns.vpn.domain.repository.UpdateRepository
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class ManifestPoller @Inject constructor(
    private val updateRepository: UpdateRepository
) {
    private val scope = CoroutineScope(Dispatchers.Default + Job())
    private var pollingJob: Job? = null

    private val _hasUpdate = MutableStateFlow(false)
    val hasUpdate: StateFlow<Boolean> = _hasUpdate.asStateFlow()

    private val _latestVersion = MutableStateFlow<String?>(null)
    val latestVersion: StateFlow<String?> = _latestVersion.asStateFlow()

    fun startPolling(intervalMs: Long = POLLING_INTERVAL_MS) {
        pollingJob?.cancel()
        pollingJob = scope.launch {
            while (isActive) {
                try {
                    val manifest = updateRepository.checkForUpdate()
                    _hasUpdate.value = manifest != null
                    _latestVersion.value = manifest?.latestVersionName
                } catch (e: Exception) {
                    android.util.Log.w("ManifestPoller", "Polling failed", e)
                }
                delay(intervalMs)
            }
        }
    }

    fun stopPolling() {
        pollingJob?.cancel()
        pollingJob = null
    }

    fun isPolling(): Boolean = pollingJob?.isActive == true

    companion object {
        private const val POLLING_INTERVAL_MS = 4 * 60 * 60 * 1000L // 4 hours
    }
}
