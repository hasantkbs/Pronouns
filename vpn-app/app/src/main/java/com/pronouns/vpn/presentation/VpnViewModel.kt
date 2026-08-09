package com.pronouns.vpn.presentation

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.pronouns.vpn.domain.model.VpnStatus
import com.pronouns.vpn.domain.repository.VpnRepository
import com.pronouns.vpn.domain.usecase.ConnectVpnUseCase
import com.pronouns.vpn.domain.usecase.RotateCredentialsUseCase
import com.pronouns.vpn.domain.usecase.VerifyTunnelUseCase
import com.pronouns.vpn.vpn.VpnStateMonitor
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import javax.inject.Inject

@HiltViewModel
class VpnViewModel @Inject constructor(
    private val vpnRepository: VpnRepository,
    private val connectVpnUseCase: ConnectVpnUseCase,
    private val verifyTunnelUseCase: VerifyTunnelUseCase,
    private val rotateCredentialsUseCase: RotateCredentialsUseCase,
    private val vpnStateMonitor: VpnStateMonitor
) : ViewModel() {

    private val _vpnStatus = MutableStateFlow<VpnStatus>(VpnStatus.Disconnected)
    val vpnStatus: StateFlow<VpnStatus> = _vpnStatus.asStateFlow()

    val connectionState = vpnStateMonitor.connectionState

    private val _trafficStats = MutableStateFlow(Pair(0L, 0L))
    val trafficStats: StateFlow<Pair<Long, Long>> = _trafficStats.asStateFlow()

    init {
        viewModelScope.launch {
            vpnRepository.vpnStatus.collect { status ->
                _vpnStatus.value = status
            }
        }
        viewModelScope.launch {
            vpnRepository.trafficStats.collect { stats ->
                _trafficStats.value = stats
            }
        }
        vpnStateMonitor.startMonitoring()
    }

    fun connect() {
        viewModelScope.launch {
            try {
                connectVpnUseCase()
            } catch (e: Exception) {
                _vpnStatus.value = VpnStatus.Error(e, e.message ?: "Connection failed")
            }
        }
    }

    fun disconnect() {
        viewModelScope.launch {
            vpnRepository.disconnect()
        }
    }

    fun reconnect() {
        viewModelScope.launch {
            vpnRepository.reconnect()
        }
    }

    fun rotateCredentials() {
        viewModelScope.launch {
            rotateCredentialsUseCase()
        }
    }

    fun verifyConnection() {
        viewModelScope.launch {
            val ok = verifyTunnelUseCase()
            if (!ok) {
                _vpnStatus.value = VpnStatus.Error(
                    RuntimeException("Tunnel compromised"),
                    "Tunnel verification failed"
                )
            }
        }
    }

    override fun onCleared() {
        super.onCleared()
        vpnStateMonitor.stopMonitoring()
    }
}
