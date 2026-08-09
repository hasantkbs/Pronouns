package com.pronouns.vpn.di

import android.content.Context
import com.pronouns.vpn.vpn.OpenVpnManager
import com.pronouns.vpn.vpn.TunnelVerifier
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.android.qualifiers.ApplicationContext
import dagger.hilt.components.SingletonComponent
import javax.inject.Singleton

@Module
@InstallIn(SingletonComponent::class)
object VpnModule {

    @Provides
    @Singleton
    fun provideOpenVpnManager(
        @ApplicationContext context: Context
    ): OpenVpnManager = OpenVpnManager(context)

    @Provides
    @Singleton
    fun provideTunnelVerifier(
        @ApplicationContext context: Context
    ): TunnelVerifier = TunnelVerifier(context)
}
