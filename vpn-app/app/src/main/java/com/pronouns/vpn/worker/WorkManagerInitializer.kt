package com.pronouns.vpn.worker

import android.content.Context
import androidx.work.WorkManager
import dagger.hilt.android.qualifiers.ApplicationContext
import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class WorkManagerInitializer @Inject constructor(
    @ApplicationContext private val context: Context
) {
    fun scheduleAll() {
        val workManager = WorkManager.getInstance(context)

        CredentialRotationWorker.schedule(workManager)
        UpdateCheckWorker.schedule(workManager)
        VpnHealthCheckWorker.schedule(workManager)

        android.util.Log.i("WorkManager", "All periodic workers scheduled")
    }

    fun cancelAll() {
        val workManager = WorkManager.getInstance(context)

        workManager.cancelUniqueWork(CredentialRotationWorker.WORK_NAME)
        workManager.cancelUniqueWork(UpdateCheckWorker.WORK_NAME)
        workManager.cancelUniqueWork(VpnHealthCheckWorker.WORK_NAME)

        android.util.Log.i("WorkManager", "All periodic workers cancelled")
    }
}

// Extension to access companion WORK_NAME from outside
internal val CredentialRotationWorker.Companion.WORK_NAME get() = "credential_rotation"
internal val UpdateCheckWorker.Companion.WORK_NAME get() = "update_check"
internal val VpnHealthCheckWorker.Companion.WORK_NAME get() = "vpn_health_check"
