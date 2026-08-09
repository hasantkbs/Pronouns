package com.pronouns.vpn.domain.usecase

import android.content.Context
import com.pronouns.vpn.core.security.SecureCredentialManager
import com.pronouns.vpn.core.utils.DeviceInfo
import com.pronouns.vpn.data.remote.ApiService
import com.pronouns.vpn.data.remote.dto.BootstrapRequest
import com.pronouns.vpn.data.repository.AuthRepository
import com.pronouns.vpn.data.repository.DeviceRepository
import com.pronouns.vpn.domain.model.VpnCredentials
import dagger.hilt.android.qualifiers.ApplicationContext
import javax.inject.Inject

data class BootstrapResult(
    val vpnCredentials: VpnCredentials,
    val authToken: String,
    val isRegistered: Boolean
)

class BootstrapDeviceUseCase @Inject constructor(
    private val authRepository: AuthRepository,
    private val deviceRepository: DeviceRepository,
    private val secureCredentialManager: SecureCredentialManager,
    private val apiService: ApiService,
    @ApplicationContext private val context: Context
) {
    suspend operator fun invoke(): Result<BootstrapResult> {
        return runCatching {
            val existingToken = authRepository.getAuthToken()

            val authToken = if (existingToken != null) {
                val healthOk = runCatching { apiService.healthCheck().isSuccessful }.getOrDefault(false)
                if (healthOk) {
                    existingToken
                } else {
                    bootstrap()
                }
            } else {
                bootstrap()
            }

            val stored = secureCredentialManager.getVpnCredentials(VPN_CREDS_ALIAS)
                ?: error("VPN credentials not available after bootstrap")

            BootstrapResult(
                vpnCredentials = VpnCredentials(stored.first, stored.second),
                authToken = authToken,
                isRegistered = existingToken != null
            )
        }
    }

    private suspend fun bootstrap(): String {
        val request = BootstrapRequest(
            deviceId = DeviceInfo.generateDeviceId(context).toString(),
            androidId = DeviceInfo.getAndroidId(context),
            manufacturer = DeviceInfo.getManufacturer(),
            model = DeviceInfo.getModel(),
            osVersion = DeviceInfo.getAndroidVersion(),
            appVersion = DeviceInfo.getAppVersion(context),
            installTimestamp = DeviceInfo.getInstallTimestamp(context),
            signatureHash = DeviceInfo.getAppSignatureHash(context)
        )
        val response = apiService.bootstrap(request)
        val token = response.authToken
            ?: error("Bootstrap response missing auth token")

        secureCredentialManager.storeAuthToken(token)
        secureCredentialManager.storeVpnCredentials(
            VPN_CREDS_ALIAS,
            response.vpnUsername,
            response.vpnPassword
        )

        return token
    }

    private companion object {
        private const val VPN_CREDS_ALIAS = "vpn_active_creds"
    }
}
