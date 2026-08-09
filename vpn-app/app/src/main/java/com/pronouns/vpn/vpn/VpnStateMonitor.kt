package com.pronouns.vpn.vpn

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.net.ConnectivityManager
import com.pronouns.vpn.domain.model.VpnStatus
import com.pronouns.vpn.data.repository.VpnRepositoryImpl
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class VpnStateMonitor @Inject constructor(
    @ApplicationContext private val context: Context,
    private val vpnRepositoryImpl: VpnRepositoryImpl,
    private val tunnelVerifier: TunnelVerifier
) {
    private val scope = CoroutineScope(Dispatchers.Default + Job())
    private var monitorJob: Job? = null

    private val _connectionState = MutableStateFlow(false)
    val connectionState: StateFlow<Boolean> = _connectionState.asStateFlow()

    private val connectivityChangeReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context, intent: Intent) {
            if (intent.action == ConnectivityManager.CONNECTIVITY_ACTION) {
                checkVpnState()
            }
        }
    }

    fun startMonitoring() {
        val filter = IntentFilter(ConnectivityManager.CONNECTIVITY_ACTION)
        context.registerReceiver(connectivityChangeReceiver, filter)

        monitorJob = scope.launch {
            while (true) {
                checkVpnState()
                delay(HEALTH_CHECK_INTERVAL_MS)
            }
        }

        tunnelVerifier.registerVpnMonitor { isVpn ->
            _connectionState.value = isVpn
            if (!isVpn) {
                vpnRepositoryImpl.updateStatus(VpnStatus.Disconnected)
            }
        }
    }

    fun stopMonitoring() {
        try {
            context.unregisterReceiver(connectivityChangeReceiver)
        } catch (e: IllegalArgumentException) {
            // Already unregistered
        }
        monitorJob?.cancel()
    }

    private fun checkVpnState() {
        val isActive = tunnelVerifier.isVpnActive()
        _connectionState.value = isActive

        if (!isActive) {
            vpnRepositoryImpl.updateStatus(VpnStatus.Disconnected)
        }
    }

    companion object {
        private const val HEALTH_CHECK_INTERVAL_MS = 30_000L
    }
}
