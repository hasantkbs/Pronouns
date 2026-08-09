package com.pronouns.vpn.domain.usecase

import com.pronouns.vpn.domain.model.UpdateManifest
import com.pronouns.vpn.domain.repository.UpdateRepository
import javax.inject.Inject

class CheckUpdateUseCase @Inject constructor(
    private val updateRepository: UpdateRepository
) {
    suspend operator fun invoke(): Result<UpdateManifest?> = runCatching {
        updateRepository.checkForUpdate()
    }
}
