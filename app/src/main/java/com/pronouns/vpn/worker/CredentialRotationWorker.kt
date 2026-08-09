package com.pronouns.vpn.worker

import android.content.Context
import androidx.hilt.work.HiltWorker
import androidx.work.CoroutineWorker
import androidx.work.Data
import androidx.work.PeriodicWorkRequest
import androidx.work.PeriodicWorkRequestBuilder
import androidx.work.WorkerParameters
import androidx.work.workDataOf
import com.pronouns.vpn.core.security.SecureCredentialManager
import com.pronouns.vpn.data.local.PreferencesStore
import com.pronouns.vpn.data.repository.AuthRepository
import com.pronouns.vpn.domain.usecase.ConnectVpnUseCase
import com.pronouns.vpn.vpn.OpenVpnManager
import dagger.assisted.Assisted
import dagger.assisted.AssistedInject
import kotlinx.coroutines.flow.first
import java.util.concurrent.TimeUnit

@HiltWorker
class CredentialRotationWorker @AssistedInject constructor(
    @Assisted appContext: Context,
    @Assisted workerParams: WorkerParameters,
    private val authRepository: AuthRepository,
    private val secureCredentialManager: SecureCredentialManager,
    private val preferencesStore: PreferencesStore,
    private val openVpnManager: OpenVpnManager,
    private val connectVpnUseCase: ConnectVpnUseCase
) : CoroutineWorker(appContext, workerParams) {

    override suspend fun doWork(): Result {
        return try {
            val lastRotation = preferencesStore.getLastVpnRotation().first() ?: 0L
            val age = System.currentTimeMillis() - lastRotation
            val ttlMillis = TTL_HOURS * 60 * 60 * 1000L

            if (age < ttlMillis * ROTATION_THRESHOLD) {
                return Result.success(outputData(rotationSuccess = false, newTtl = ttlMillis))
            }

            val authResult = authRepository.bootstrapDevice()
            if (authResult.isFailure) return Result.failure()

            val token = authResult.getOrThrow()
            secureCredentialManager.rotateVpnCredentials(
                "vpn_rotated_${System.currentTimeMillis()}"
            )

            if (openVpnManager.isConnected()) {
                openVpnManager.disconnect()
                connectVpnUseCase()
            }

            val now = System.currentTimeMillis()
            preferencesStore.saveLastVpnRotation(now)

            Result.success(outputData(rotationSuccess = true, newTtl = token.expiry))
        } catch (_: Exception) {
            Result.failure()
        }
    }

    private fun outputData(rotationSuccess: Boolean, newTtl: Long): Data {
        return workDataOf(
            KEY_ROTATION_SUCCESS to rotationSuccess,
            KEY_NEW_TTL to newTtl
        )
    }

    companion object {
        const val TTL_HOURS = 6L
        const val FLEX_HOURS = 1L
        const val KEY_ROTATION_SUCCESS = "rotation_success"
        const val KEY_NEW_TTL = "new_ttl"
        private const val ROTATION_THRESHOLD = 0.7

        fun createPeriodicRequest(): PeriodicWorkRequest {
            return PeriodicWorkRequestBuilder<CredentialRotationWorker>(
                TTL_HOURS, TimeUnit.HOURS,
                FLEX_HOURS, TimeUnit.HOURS
            ).build()
        }
    }
}
