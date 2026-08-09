package com.pronouns.vpn.domain.usecase

import com.pronouns.vpn.domain.repository.AuthRepository
import com.pronouns.vpn.domain.repository.VpnRepository
import javax.inject.Inject

class RotateCredentialsUseCase @Inject constructor(
    private val authRepository: AuthRepository,
    private val vpnRepository: VpnRepository
) {
    suspend operator fun invoke(): Result<Unit> = runCatching {
        val newCredentials = authRepository.rotateCredentials()
        if (vpnRepository.isConnected()) {
            vpnRepository.reconnect()
        }
    }
}
