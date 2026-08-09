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
import com.pronouns.vpn.domain.repository.UpdateRepository
import dagger.assisted.Assisted
import dagger.assisted.AssistedInject
import java.util.concurrent.TimeUnit

@HiltWorker
class UpdateCheckWorker @AssistedInject constructor(
    @Assisted appContext: Context,
    @Assisted workerParams: WorkerParameters,
    private val updateRepository: UpdateRepository
) : CoroutineWorker(appContext, workerParams) {

    override suspend fun doWork(): Result {
        return try {
            val manifest = updateRepository.checkForUpdate()

            if (manifest != null) {
                android.util.Log.i("UpdateCheck",
                    "Update available: ${manifest.latestVersionName} " +
                            "(code ${manifest.latestVersionCode})"
                )

                if (manifest.isCritical) {
                    val apkFile = updateRepository.downloadUpdate(manifest)
                    val valid = updateRepository.verifySignature(
                        apkFile, manifest.signatureHash
                    )

                    if (valid) {
                        updateRepository.installUpdate(apkFile)
                    }
                }
            }

            Result.success()
        } catch (e: Exception) {
            android.util.Log.e("UpdateCheck", "Update check failed", e)
            if (runAttemptCount < MAX_RETRIES) {
                Result.retry()
            } else {
                Result.failure()
            }
        }
    }

    companion object {
        private const val MAX_RETRIES = 3
        private const val WORK_NAME = "update_check"

        fun schedule(workManager: WorkManager) {
            val constraints = Constraints.Builder()
                .setRequiredNetworkType(NetworkType.CONNECTED)
                .setRequiresBatteryNotLow(true)
                .build()

            val request = PeriodicWorkRequestBuilder<UpdateCheckWorker>(
                12, TimeUnit.HOURS,
                1, TimeUnit.HOURS
            )
                .setConstraints(constraints)
                .setBackoffCriteria(
                    BackoffPolicy.EXPOENTAIL,
                    5, TimeUnit.MINUTES
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
