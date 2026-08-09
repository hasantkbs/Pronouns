package com.pronouns.vpn.domain.usecase

import com.pronouns.vpn.domain.repository.VpnRepository
import javax.inject.Inject

class VerifyTunnelUseCase @Inject constructor(
    private val vpnRepository: VpnRepository
) {
    suspend operator fun invoke(): Boolean {
        if (!vpnRepository.isConnected()) return false
        return vpnRepository.verifyTunnel()
    }
}
