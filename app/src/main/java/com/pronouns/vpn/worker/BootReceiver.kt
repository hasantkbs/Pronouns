package com.pronouns.vpn.worker

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import androidx.work.ExistingPeriodicWorkPolicy
import androidx.work.WorkManager
import com.pronouns.vpn.data.local.PreferencesStore
import com.pronouns.vpn.domain.usecase.ConnectVpnUseCase
import dagger.hilt.android.AndroidEntryPoint
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.launch
import javax.inject.Inject

@AndroidEntryPoint
class BootReceiver : BroadcastReceiver() {

    @Inject lateinit var preferencesStore: PreferencesStore
    @Inject lateinit var connectVpnUseCase: ConnectVpnUseCase

    override fun onReceive(context: Context, intent: Intent) {
        if (intent.action != Intent.ACTION_BOOT_COMPLETED) return

        val workManager = WorkManager.getInstance(context)

        workManager.enqueueUniquePeriodicWork(
            WORK_NAME_CREDENTIAL_ROTATION,
            ExistingPeriodicWorkPolicy.KEEP,
            CredentialRotationWorker.createPeriodicRequest()
        )

        workManager.enqueueUniquePeriodicWork(
            WORK_NAME_UPDATE_POLL,
            ExistingPeriodicWorkPolicy.KEEP,
            UpdatePollWorker.createPeriodicRequest()
        )

        workManager.enqueueUniquePeriodicWork(
            WORK_NAME_VPN_HEALTH_CHECK,
            ExistingPeriodicWorkPolicy.KEEP,
            VpnHealthCheckWorker.createPeriodicRequest()
        )

        val pendingResult = goAsync()
        CoroutineScope(Dispatchers.IO).launch {
            try {
                val lastState = preferencesStore.getLastVpnState().first()
                if (lastState == VPN_STATE_CONNECTED) {
                    connectVpnUseCase()
                }
            } finally {
                pendingResult.finish()
            }
        }
    }

    companion object {
        private const val WORK_NAME_CREDENTIAL_ROTATION = "credential_rotation"
        private const val WORK_NAME_UPDATE_POLL = "update_poll"
        private const val WORK_NAME_VPN_HEALTH_CHECK = "vpn_health_check"
        private const val VPN_STATE_CONNECTED = "CONNECTED"
    }
}
