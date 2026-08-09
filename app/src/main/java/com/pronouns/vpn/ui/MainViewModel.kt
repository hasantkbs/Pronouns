package com.pronouns.vpn.ui

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import androidx.work.ExistingPeriodicWorkPolicy
import androidx.work.WorkInfo
import androidx.work.WorkManager
import com.pronouns.vpn.domain.usecase.BootstrapDeviceUseCase
import com.pronouns.vpn.domain.usecase.ConnectVpnUseCase
import com.pronouns.vpn.vpn.TunnelVerifier
import com.pronouns.vpn.worker.CredentialRotationWorker
import com.pronouns.vpn.worker.UpdatePollWorker
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.catch
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.launch
import java.util.concurrent.TimeUnit
import javax.inject.Inject

enum class VpnConnectionState {
    DISCONNECTED,
    CONNECTING,
    CONNECTED
}

@HiltViewModel
class MainViewModel @Inject constructor(
    private val bootstrapDeviceUseCase: BootstrapDeviceUseCase,
    private val connectVpnUseCase: ConnectVpnUseCase,
    private val tunnelVerifier: TunnelVerifier,
    private val workManager: WorkManager
) : ViewModel() {

    private val _connectionState = MutableStateFlow(VpnConnectionState.DISCONNECTED)
    val connectionState: StateFlow<VpnConnectionState> = _connectionState.asStateFlow()

    private val _errorState = MutableStateFlow<String?>(null)
    val errorState: StateFlow<String?> = _errorState.asStateFlow()

    private val _updateAvailable = MutableStateFlow(false)
    val updateAvailable: StateFlow<Boolean> = _updateAvailable.asStateFlow()

    private val _healthStatus = MutableStateFlow("Unknown")
    val healthStatus: StateFlow<String> = _healthStatus.asStateFlow()

    init {
        scheduleCredentialRotation()
        scheduleUpdatePolling()
    }

    fun onConnect() {
        viewModelScope.launch {
            _connectionState.value = VpnConnectionState.CONNECTING
            _errorState.value = null
            try {
                bootstrapDeviceUseCase()
                connectVpnUseCase()
                tunnelVerifier.verify()
                _connectionState.value = VpnConnectionState.CONNECTED
            } catch (e: Exception) {
                _errorState.value = e.message ?: "Connection failed"
                _connectionState.value = VpnConnectionState.DISCONNECTED
            }
        }
    }

    fun onDisconnect() {
        viewModelScope.launch {
            try {
                connectVpnUseCase.disconnect()
            } finally {
                _connectionState.value = VpnConnectionState.DISCONNECTED
            }
        }
    }

    fun installUpdate() {
        _updateAvailable.value = false
    }

    private fun scheduleCredentialRotation() {
        val request = CredentialRotationWorker.createPeriodicRequest()

        workManager.enqueueUniquePeriodicWork(
            "credential_rotation",
            ExistingPeriodicWorkPolicy.KEEP,
            request
        )
    }

    private fun scheduleUpdatePolling() {
        val request = UpdatePollWorker.createPeriodicRequest()

        workManager.enqueueUniquePeriodicWork(
            "update_polling",
            ExistingPeriodicWorkPolicy.KEEP,
            request
        )

        viewModelScope.launch {
            workManager.getWorkInfoByIdFlow(request.id)
                .catch { _updateAvailable.value = false }
                .collectLatest { info ->
                    when (info.state) {
                        WorkInfo.State.SUCCEEDED -> {
                            _updateAvailable.value = info.outputData
                                .getBoolean("update_available", false)
                        }
                        WorkInfo.State.FAILED -> {
                            _updateAvailable.value = false
                        }
                        else -> {}
                    }
                }
        }
    }
}
