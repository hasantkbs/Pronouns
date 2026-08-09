package com.pronouns.vpn.domain.usecase

import com.pronouns.vpn.data.remote.api.BootstrapApi
import com.pronouns.vpn.data.remote.dto.BootstrapRequest
import com.pronouns.vpn.domain.model.DeviceInfo
import com.pronouns.vpn.domain.repository.AuthRepository
import com.pronouns.vpn.domain.repository.DeviceRepository
import com.pronouns.vpn.security.DetectorManager
import javax.inject.Inject

class BootstrapDeviceUseCase @Inject constructor(
    private val bootstrapApi: BootstrapApi,
    private val deviceRepository: DeviceRepository,
    private val authRepository: AuthRepository,
    private val detectorManager: DetectorManager
) {
    suspend operator fun invoke(): Result<Unit> = runCatching {
        if (detectorManager.isCompromised()) {
            throw SecurityException("Device security check failed")
        }

        val deviceInfo = deviceRepository.getDeviceInfo()
        val deviceId = deviceRepository.getDeviceId()

        val request = BootstrapRequest(
            deviceId = deviceId,
            androidId = deviceInfo.androidId,
            manufacturer = deviceInfo.manufacturer,
            model = deviceInfo.model,
            androidVersion = deviceInfo.androidVersion,
            appVersion = deviceInfo.appVersion,
            installTimestamp = deviceInfo.installTimestamp,
            appSignatureHash = deviceInfo.appSignatureHash
        )

        val response = bootstrapApi.bootstrap(request)

        authRepository.saveCredentials(
            com.pronouns.vpn.domain.model.AuthCredentials(
                username = response.vpnUsername,
                password = response.vpnPassword,
                expiresAt = java.time.Instant.ofEpochSecond(response.expiresAt),
                isEphemeral = true
            )
        )

        response.authToken?.let { authRepository.saveApiToken(it) }

        if (response.deviceRegistered) {
            deviceRepository.markDeviceRegistered()
        }
    }
}
