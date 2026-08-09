package com.pronouns.vpn.di

import com.pronouns.vpn.security.AttestationValidator
import com.pronouns.vpn.security.CertificatePinner
import com.pronouns.vpn.security.DebuggerDetector
import com.pronouns.vpn.security.DetectorManager
import com.pronouns.vpn.security.EmulatorDetector
import com.pronouns.vpn.security.FridaDetector
import com.pronouns.vpn.security.RootDetector
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.components.SingletonComponent
import javax.inject.Singleton

@Module
@InstallIn(SingletonComponent::class)
object SecurityModule {

    @Provides
    @Singleton
    fun provideRootDetector(): RootDetector = RootDetector()

    @Provides
    @Singleton
    fun provideDebuggerDetector(): DebuggerDetector = DebuggerDetector()

    @Provides
    @Singleton
    fun provideEmulatorDetector(): EmulatorDetector = EmulatorDetector()

    @Provides
    @Singleton
    fun provideFridaDetector(): FridaDetector = FridaDetector()

    @Provides
    @Singleton
    fun provideCertificatePinner(): CertificatePinner = CertificatePinner()

    @Provides
    @Singleton
    fun provideAttestationValidator(): AttestationValidator = AttestationValidator()

    @Provides
    @Singleton
    fun provideDetectorManager(
        rootDetector: RootDetector,
        debuggerDetector: DebuggerDetector,
        emulatorDetector: EmulatorDetector,
        fridaDetector: FridaDetector
    ): DetectorManager = DetectorManager(
        rootDetector = rootDetector,
        debuggerDetector = debuggerDetector,
        emulatorDetector = emulatorDetector,
        fridaDetector = fridaDetector
    )
}
