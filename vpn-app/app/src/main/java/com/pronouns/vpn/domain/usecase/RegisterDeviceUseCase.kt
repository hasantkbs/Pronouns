package com.pronouns.vpn.domain.usecase

import com.pronouns.vpn.data.remote.api.DeviceApi
import com.pronouns.vpn.data.remote.dto.DeviceRegistrationRequest
import com.pronouns.vpn.domain.repository.AuthRepository
import com.pronouns.vpn.domain.repository.DeviceRepository
import javax.inject.Inject

class RegisterDeviceUseCase @Inject constructor(
    private val deviceApi: DeviceApi,
    private val deviceRepository: DeviceRepository,
    private val authRepository: AuthRepository
) {
    suspend operator fun invoke(): Result<String> = runCatching {
        val deviceInfo = deviceRepository.getDeviceInfo()
        val deviceId = deviceRepository.getDeviceId()

        val request = DeviceRegistrationRequest(
            deviceId = deviceId,
            androidId = deviceInfo.androidId,
            manufacturer = deviceInfo.manufacturer,
            model = deviceInfo.model,
            androidVersion = deviceInfo.androidVersion,
            appVersion = deviceInfo.appVersion,
            installTimestamp = deviceInfo.installTimestamp,
            appSignatureHash = deviceInfo.appSignatureHash
        )

        val response = deviceApi.registerDevice(
            authToken = authRepository.getApiToken() ?: "",
            request = request
        )

        authRepository.saveApiToken(response.authToken)

        if (response.featureFlags.isNotEmpty()) {
            response.featureFlags.forEach { (key, value) ->
                android.util.Log.d("RegisterDevice", "Feature flag: $key = $value")
            }
        }

        deviceRepository.markDeviceRegistered()
        response.authToken
    }
}
