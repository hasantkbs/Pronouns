package com.pronouns.vpn.presentation

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.pronouns.vpn.domain.repository.AuthRepository
import com.pronouns.vpn.domain.repository.DeviceRepository
import com.pronouns.vpn.domain.usecase.BootstrapDeviceUseCase
import com.pronouns.vpn.domain.usecase.ConnectVpnUseCase
import com.pronouns.vpn.domain.usecase.RegisterDeviceUseCase
import com.pronouns.vpn.domain.usecase.VerifyTunnelUseCase
import com.pronouns.vpn.security.DetectorManager
import com.pronouns.vpn.worker.WorkManagerInitializer
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import javax.inject.Inject

@HiltViewModel
class MainViewModel @Inject constructor(
    application: Application,
    private val bootstrapDeviceUseCase: BootstrapDeviceUseCase,
    private val connectVpnUseCase: ConnectVpnUseCase,
    private val registerDeviceUseCase: RegisterDeviceUseCase,
    private val verifyTunnelUseCase: VerifyTunnelUseCase,
    private val authRepository: AuthRepository,
    private val deviceRepository: DeviceRepository,
    private val detectorManager: DetectorManager,
    private val workManagerInitializer: WorkManagerInitializer
) : AndroidViewModel(application) {

    sealed interface UiState {
        data object Initializing : UiState
        data object SecurityCheck : UiState
        data class SecurityFailed(val reasons: List<String>) : UiState
        data object Bootstrapping : UiState
        data object ConnectingVpn : UiState
        data object VerifyingTunnel : UiState
        data object RegisteringDevice : UiState
        data object Ready : UiState
        data class Error(val message: String) : UiState
    }

    private val _uiState = MutableStateFlow<UiState>(UiState.Initializing)
    val uiState: StateFlow<UiState> = _uiState.asStateFlow()

    fun start() {
        viewModelScope.launch {
            executeStartupFlow()
        }
    }

    private suspend fun executeStartupFlow() {
        _uiState.value = UiState.SecurityCheck

        val securityStatus = detectorManager.getSecurityStatus()
        if (securityStatus.isCompromised) {
            val reasons = buildList {
                if (securityStatus.isRooted) add("Device is rooted")
                if (securityStatus.isDebuggerAttached) add("Debugger detected")
                if (securityStatus.isEmulator) add("Emulator detected")
                if (securityStatus.isFridaDetected) add("Frida detected")
            }
            _uiState.value = UiState.SecurityFailed(reasons)
            return
        }

        if (!authRepository.hasCredentials()) {
            _uiState.value = UiState.Bootstrapping
            val bootstrapResult = bootstrapDeviceUseCase()
            if (bootstrapResult.isFailure) {
                _uiState.value = UiState.Error(
                    bootstrapResult.exceptionOrNull()?.message ?: "Bootstrap failed"
                )
                return
            }
        }

        _uiState.value = UiState.ConnectingVpn
        try {
            connectVpnUseCase()
        } catch (e: Exception) {
            _uiState.value = UiState.Error("VPN connection failed: ${e.message}")
            return
        }

        _uiState.value = UiState.VerifyingTunnel
        val tunnelOk = verifyTunnelUseCase()
        if (!tunnelOk) {
            _uiState.value = UiState.Error("Tunnel verification failed")
            return
        }

        if (!deviceRepository.isDeviceRegistered() && authRepository.hasApiToken()) {
            _uiState.value = UiState.RegisteringDevice
            val registerResult = registerDeviceUseCase()
            if (registerResult.isFailure) {
                android.util.Log.w("MainViewModel",
                    "Device registration failed: ${registerResult.exceptionOrNull()?.message}"
                )
            }
        }

        workManagerInitializer.scheduleAll()
        _uiState.value = UiState.Ready
    }

    fun retry() {
        _uiState.value = UiState.Initializing
        start()
    }
}
