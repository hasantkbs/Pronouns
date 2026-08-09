package com.pronouns.vpn.worker

import android.content.Context
import androidx.hilt.work.HiltWorker
import androidx.work.BackoffPolicy
import androidx.work.Constraints
import androidx.work.CoroutineWorker
import androidx.work.ExistingPeriodicWorkPolicy
import androidx.work.NetworkType
import androidx.work.PeriodicWorkRequestBuilder
import androidx.work.WorkManager
import androidx.work.WorkerParameters
import com.pronouns.vpn.domain.repository.AuthRepository
import com.pronouns.vpn.domain.repository.VpnRepository
import dagger.assisted.Assisted
import dagger.assisted.AssistedInject
import java.util.concurrent.TimeUnit

@HiltWorker
class CredentialRotationWorker @AssistedInject constructor(
    @Assisted appContext: Context,
    @Assisted workerParams: WorkerParameters,
    private val authRepository: AuthRepository,
    private val vpnRepository: VpnRepository
) : CoroutineWorker(appContext, workerParams) {

    override suspend fun doWork(): Result {
        return try {
            val credentials = authRepository.getStoredCredentials()

            if (credentials != null && credentials.isExpiringWithin(EXPIRY_THRESHOLD_SECONDS)) {
                android.util.Log.i("CredentialRotation", "Rotating VPN credentials")

                val newCredentials = authRepository.rotateCredentials()

                if (vpnRepository.isConnected()) {
                    vpnRepository.reconnect()
                }

                android.util.Log.i("CredentialRotation", "Credentials rotated successfully")
            }

            Result.success()
        } catch (e: Exception) {
            android.util.Log.e("CredentialRotation", "Rotation failed", e)
            if (runAttemptCount < MAX_RETRIES) {
                Result.retry()
            } else {
                Result.failure()
            }
        }
    }

    companion object {
        private const val EXPIRY_THRESHOLD_SECONDS = 300L
        private const val MAX_RETRIES = 3
        private const val WORK_NAME = "credential_rotation"

        fun schedule(workManager: WorkManager) {
            val constraints = Constraints.Builder()
                .setRequiredNetworkType(NetworkType.CONNECTED)
                .build()

            val request = PeriodicWorkRequestBuilder<CredentialRotationWorker>(
                4, TimeUnit.HOURS,
                15, TimeUnit.MINUTES
            )
                .setConstraints(constraints)
                .setBackoffCriteria(
                    BackoffPolicy.EXPONENTIAL,
                    1, TimeUnit.MINUTES
                )
                .build()

            workManager.enqueueUniquePeriodicWork(
                WORK_NAME,
                ExistingPeriodicWorkPolicy.KEEP,
                request
            )
        }
    }
}
