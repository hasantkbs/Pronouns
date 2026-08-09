package com.pronouns.vpn.domain.usecase

import com.pronouns.vpn.domain.repository.AuthRepository
import com.pronouns.vpn.domain.repository.VpnRepository
import javax.inject.Inject

class ConnectVpnUseCase @Inject constructor(
    private val vpnRepository: VpnRepository,
    private val authRepository: AuthRepository
) {
    suspend operator fun invoke() {
        val credentials = authRepository.getStoredCredentials()
            ?: throw IllegalStateException("No VPN credentials available")

        if (credentials.isExpired()) {
            val rotated = authRepository.rotateCredentials()
            vpnRepository.connect(rotated.username, rotated.password)
            return
        }

        vpnRepository.connect(credentials.username, credentials.password)
    }
}
