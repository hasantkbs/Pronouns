package com.pronouns.vpn.worker

import android.app.NotificationManager
import android.content.Context
import android.content.pm.PackageManager
import android.os.Build
import androidx.core.app.NotificationCompat
import androidx.hilt.work.HiltWorker
import androidx.work.CoroutineWorker
import androidx.work.PeriodicWorkRequest
import androidx.work.PeriodicWorkRequestBuilder
import androidx.work.WorkerParameters
import com.pronouns.vpn.PronounsApplication
import com.pronouns.vpn.data.repository.UpdateRepository
import dagger.assisted.Assisted
import dagger.assisted.AssistedInject
import java.util.concurrent.TimeUnit

@HiltWorker
class UpdatePollWorker @AssistedInject constructor(
    @Assisted appContext: Context,
    @Assisted workerParams: WorkerParameters,
    private val updateRepository: UpdateRepository
) : CoroutineWorker(appContext, workerParams) {

    override suspend fun doWork(): Result {
        return try {
            val manifestResult = updateRepository.checkForUpdates()
            if (manifestResult.isFailure) return Result.success()

            val manifest = manifestResult.getOrThrow()
            if (manifest.versionCode <= getCurrentVersionCode()) return Result.success()

            val downloadResult = updateRepository.downloadUpdate(manifest)
            if (downloadResult.isFailure) return Result.failure()

            val file = downloadResult.getOrThrow()
            val verified = updateRepository.verifyUpdateSignature(file, manifest.signatureHash)
            if (!verified) return Result.failure()

            val installResult = updateRepository.installUpdate(file)
            if (installResult.isFailure) return Result.failure()

            if (manifest.required) {
                showUpdateNotification()
            }

            Result.success()
        } catch (_: Exception) {
            Result.failure()
        }
    }

    @Suppress("DEPRECATION")
    private fun getCurrentVersionCode(): Int {
        val pm = applicationContext.packageManager
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            val info = pm.getPackageInfo(
                applicationContext.packageName,
                PackageManager.PackageInfoFlags.of(0L)
            )
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
                info.longVersionCode.toInt()
            } else {
                info.versionCode
            }
        } else {
            val info = pm.getPackageInfo(applicationContext.packageName, 0)
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
                info.longVersionCode.toInt()
            } else {
                info.versionCode
            }
        }
    }

    private fun showUpdateNotification() {
        val notificationManager = applicationContext
            .getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        val notification = NotificationCompat.Builder(
            applicationContext,
            PronounsApplication.UPDATE_CHANNEL_ID
        )
            .setSmallIcon(android.R.drawable.ic_dialog_info)
            .setContentTitle("Update Required")
            .setContentText("A critical update is available. Please install it now.")
            .setPriority(NotificationCompat.PRIORITY_HIGH)
            .setAutoCancel(true)
            .build()
        notificationManager.notify(NOTIFICATION_ID, notification)
    }

    companion object {
        const val INTERVAL_HOURS = 4L
        private const val NOTIFICATION_ID = 1001

        fun createPeriodicRequest(): PeriodicWorkRequest {
            return PeriodicWorkRequestBuilder<UpdatePollWorker>(
                INTERVAL_HOURS, TimeUnit.HOURS
            ).build()
        }
    }
}
