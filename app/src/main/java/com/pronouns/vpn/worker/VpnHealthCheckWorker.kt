package com.pronouns.vpn.worker

import android.app.NotificationManager
import android.content.Context
import androidx.core.app.NotificationCompat
import androidx.hilt.work.HiltWorker
import androidx.work.CoroutineWorker
import androidx.work.OneTimeWorkRequest
import androidx.work.OneTimeWorkRequestBuilder
import androidx.work.PeriodicWorkRequest
import androidx.work.PeriodicWorkRequestBuilder
import androidx.work.WorkerParameters
import androidx.work.workDataOf
import com.pronouns.vpn.PronounsApplication
import com.pronouns.vpn.domain.usecase.ConnectVpnUseCase
import com.pronouns.vpn.vpn.OpenVpnManager
import com.pronouns.vpn.vpn.TunnelVerifier
import dagger.assisted.Assisted
import dagger.assisted.AssistedInject
import kotlinx.coroutines.delay
import java.util.concurrent.TimeUnit

@HiltWorker
class VpnHealthCheckWorker @AssistedInject constructor(
    @Assisted appContext: Context,
    @Assisted workerParams: WorkerParameters,
    private val openVpnManager: OpenVpnManager,
    private val tunnelVerifier: TunnelVerifier,
    private val connectVpnUseCase: ConnectVpnUseCase
) : CoroutineWorker(appContext, workerParams) {

    override suspend fun doWork(): Result {
        val maxRetries = inputData.getInt(KEY_MAX_RETRIES, DEFAULT_MAX_RETRIES)
        return try {
            if (!openVpnManager.isConnected()) return Result.success()

            val tunnelOk = tunnelVerifier.verifyTunnel()
            val backendOk = tunnelVerifier.verifyBackendReachable(BACKEND_URL)

            if (tunnelOk && backendOk) return Result.success()

            reconnectWithRetries(maxRetries)
        } catch (_: Exception) {
            reconnectWithRetries(maxRetries)
        }
    }

    private suspend fun reconnectWithRetries(maxRetries: Int): Result {
        for (attempt in 1..maxRetries) {
            val result = connectVpnUseCase()
            if (result.isSuccess) return Result.success()
            delay(RETRY_DELAY_MS)
        }
        showCriticalErrorNotification()
        return Result.failure()
    }

    private fun showCriticalErrorNotification() {
        val notificationManager = applicationContext
            .getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        val notification = NotificationCompat.Builder(
            applicationContext,
            PronounsApplication.VPN_CHANNEL_ID
        )
            .setSmallIcon(android.R.drawable.ic_dialog_alert)
            .setContentTitle("VPN Connection Lost")
            .setContentText("Unable to reconnect after multiple attempts. Please open the app.")
            .setPriority(NotificationCompat.PRIORITY_MAX)
            .setAutoCancel(true)
            .build()
        notificationManager.notify(NOTIFICATION_ID, notification)
    }

    companion object {
        const val KEY_MAX_RETRIES = "max_retries"
        const val PERIODIC_INTERVAL_MINUTES = 15L
        private const val DEFAULT_MAX_RETRIES = 3
        private const val BACKEND_URL = "https://api.pronouns.vpn/api/v1/health"
        private const val RETRY_DELAY_MS = 5000L
        private const val NOTIFICATION_ID = 1002

        fun createPeriodicRequest(): PeriodicWorkRequest {
            return PeriodicWorkRequestBuilder<VpnHealthCheckWorker>(
                PERIODIC_INTERVAL_MINUTES, TimeUnit.MINUTES
            ).build()
        }

        fun createOneTimeRequest(maxRetries: Int = DEFAULT_MAX_RETRIES): OneTimeWorkRequest {
            return OneTimeWorkRequestBuilder<VpnHealthCheckWorker>()
                .setInputData(workDataOf(KEY_MAX_RETRIES to maxRetries))
                .build()
        }
    }
}
