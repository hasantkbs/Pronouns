package com.pronouns.vpn.di

import com.pronouns.vpn.BuildConfig
import com.pronouns.vpn.data.remote.ApiService
import com.pronouns.vpn.data.remote.interceptors.AuthInterceptor
import com.pronouns.vpn.data.remote.interceptors.DeviceAttestationInterceptor
import com.squareup.moshi.Moshi
import com.squareup.moshi.kotlin.reflect.KotlinJsonAdapterFactory
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.components.SingletonComponent
import okhttp3.ConnectionPool
import okhttp3.ConnectionSpec
import okhttp3.OkHttpClient
import okhttp3.TlsVersion
import okhttp3.logging.HttpLoggingInterceptor
import retrofit2.Retrofit
import retrofit2.converter.moshi.MoshiConverterFactory
import java.util.concurrent.TimeUnit
import javax.inject.Qualifier
import javax.inject.Singleton

@Qualifier
@Retention(AnnotationRetention.BINARY)
annotation class Json

@Module
@InstallIn(SingletonComponent::class)
object NetworkModule {

    @Provides
    @Singleton
    fun provideHttpLoggingInterceptor(): HttpLoggingInterceptor {
        val level = if (BuildConfig.DEBUG) {
            HttpLoggingInterceptor.Level.BODY
        } else {
            HttpLoggingInterceptor.Level.NONE
        }
        return HttpLoggingInterceptor().apply { setLevel(level) }
    }

    @Provides
    @Singleton
    fun provideConnectionSpecs(): List<ConnectionSpec> {
        return listOf(
            ConnectionSpec.Builder(ConnectionSpec.MODERN_TLS)
                .tlsVersions(TlsVersion.TLS_1_3, TlsVersion.TLS_1_2)
                .build(),
            ConnectionSpec.CLEARTEXT
        )
    }

    @Provides
    @Singleton
    fun provideOkHttpClient(
        certificatePinner: okhttp3.CertificatePinner,
        authInterceptor: AuthInterceptor,
        deviceAttestationInterceptor: DeviceAttestationInterceptor,
        loggingInterceptor: HttpLoggingInterceptor,
        connectionSpecs: List<ConnectionSpec>
    ): OkHttpClient {
        return OkHttpClient.Builder()
            .certificatePinner(certificatePinner)
            .addInterceptor(authInterceptor)
            .addInterceptor(deviceAttestationInterceptor)
            .addInterceptor(loggingInterceptor)
            .connectionSpecs(connectionSpecs)
            .connectTimeout(30, TimeUnit.SECONDS)
            .readTimeout(30, TimeUnit.SECONDS)
            .writeTimeout(30, TimeUnit.SECONDS)
            .connectionPool(ConnectionPool(5, 30, TimeUnit.SECONDS))
            .followRedirects(true)
            .followSslRedirects(true)
            .build()
    }

    @Provides
    @Singleton
    @Json
    fun provideMoshi(): Moshi {
        return Moshi.Builder()
            .addLast(KotlinJsonAdapterFactory())
            .build()
    }

    @Provides
    @Singleton
    fun provideRetrofit(
        okHttpClient: OkHttpClient,
        @Json moshi: Moshi
    ): Retrofit {
        val baseUrl = try {
            BuildConfig.API_BASE_URL
        } catch (_: Exception) {
            "https://api.pronouns.vpn"
        }
        return Retrofit.Builder()
            .baseUrl(baseUrl)
            .client(okHttpClient)
            .addConverterFactory(MoshiConverterFactory.create(moshi))
            .build()
    }

    @Provides
    @Singleton
    fun provideApiService(retrofit: Retrofit): ApiService {
        return retrofit.create(ApiService::class.java)
    }

}
