package com.pronouns.vpn.di

import com.pronouns.vpn.vpn.OpenVpnManager
import com.pronouns.vpn.vpn.OpenVpnManagerImpl
import dagger.Binds
import dagger.Module
import dagger.hilt.InstallIn
import dagger.hilt.components.SingletonComponent
import javax.inject.Singleton

@Module
@InstallIn(SingletonComponent::class)
abstract class VpnModule {

    @Binds
    @Singleton
    abstract fun bindOpenVpnManager(
        impl: OpenVpnManagerImpl
    ): OpenVpnManager
}
