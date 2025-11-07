import streamlit as st
import numpy as np
import plotly.graph_objects as go
from io import BytesIO
import wave
import struct
import base64

# Настройка страницы
st.set_page_config(
    page_title="Генератор Розового Шума для Медитации",
    page_icon="🎵",
    layout="wide"
)

class PinkNoiseGenerator:
    """Класс для генерации розового шума с характеристикой 1/f"""
    
    def __init__(self, duration=10, sample_rate=44100):
        self.duration = duration
        self.sample_rate = sample_rate
        self.noise = None
        
    def generate_pink_noise_simple(self):
        """
        Упрощенная генерация розового шума методом суммирования октав
        """
        samples = int(self.duration * self.sample_rate)
        pink_noise = np.zeros(samples)
        num_octaves = 7
        
        for octave in range(num_octaves):
            frequency = 2 ** octave
            # Генерируем белый шум для этой октавы
            white = np.random.randn(samples // frequency + 1)
            
            # Интерполируем до нужного размера
            indices = np.arange(len(white)) * frequency
            indices = indices[:samples // frequency + 1]
            
            # Простая интерполяция
            for i in range(len(indices) - 1):
                start_idx = indices[i]
                end_idx = min(indices[i + 1], samples)
                if start_idx < samples:
                    value = white[i]
                    pink_noise[start_idx:end_idx] += value / (octave + 1)
        
        # Нормализация
        if np.max(np.abs(pink_noise)) > 0:
            pink_noise = pink_noise / np.max(np.abs(pink_noise))
        
        self.noise = pink_noise
        return pink_noise
    
    def generate_pink_noise_fft(self, alpha=1.0):
        """
        Генерация розового шума через FFT
        """
        samples = int(self.duration * self.sample_rate)
        
        # Генерируем белый шум
        white_noise = np.random.randn(samples)
        
        # FFT белого шума
        fft_white = np.fft.rfft(white_noise)
        
        # Создаем фильтр 1/f^alpha
        frequencies = np.fft.rfftfreq(samples, 1/self.sample_rate)
        # Избегаем деления на ноль
        frequencies[0] = 1
        pink_filter = 1 / np.sqrt(frequencies ** alpha)
        pink_filter[0] = pink_filter[1]
        
        # Применяем фильтр
        fft_pink = fft_white * pink_filter
        
        # Обратное FFT
        pink_noise = np.fft.irfft(fft_pink, n=samples)
        
        # Нормализация
        if np.max(np.abs(pink_noise)) > 0:
            pink_noise = pink_noise / np.max(np.abs(pink_noise))
        
        self.noise = pink_noise
        return pink_noise
    
    def apply_envelope(self, fade_in=1.0, fade_out=1.0):
        """Применение огибающей для плавного начала и конца"""
        if self.noise is None:
            return
        
        samples = len(self.noise)
        fade_in_samples = int(fade_in * self.sample_rate)
        fade_out_samples = int(fade_out * self.sample_rate)
        
        # Ограничиваем размер fade чтобы не превысить длину сигнала
        fade_in_samples = min(fade_in_samples, samples // 4)
        fade_out_samples = min(fade_out_samples, samples // 4)
        
        if fade_in_samples > 0:
            fade_in_curve = np.linspace(0, 1, fade_in_samples)
            self.noise[:fade_in_samples] *= fade_in_curve
        
        if fade_out_samples > 0:
            fade_out_curve = np.linspace(1, 0, fade_out_samples)
            self.noise[-fade_out_samples:] *= fade_out_curve
    
    def save_wav(self, volume=0.5):
        """Сохранение в WAV файл"""
        if self.noise is None:
            return None
        
        buffer = BytesIO()
        
        # Применяем громкость и конвертируем в 16-bit
        audio_data = (self.noise * volume * 32767).astype(np.int16)
        
        # Создаем WAV файл
        with wave.open(buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(audio_data.tobytes())
        
        buffer.seek(0)
        return buffer
    
    def get_audio_html(self, volume=0.5):
        """Получить HTML5 audio элемент"""
        wav_buffer = self.save_wav(volume)
        if wav_buffer:
            audio_base64 = base64.b64encode(wav_buffer.read()).decode()
            audio_html = f"""
            <audio controls style="width: 100%;">
                <source src="data:audio/wav;base64,{audio_base64}" type="audio/wav">
                Ваш браузер не поддерживает аудио элемент.
            </audio>
            """
            return audio_html
        return None

def main():
    st.title("🎵 Генератор Розового Шума для Медитации")
    st.markdown("""
    **Розовый шум** - природный звук с частотной характеристикой 1/f, 
    идеальный для медитации, концентрации и улучшения сна.
    """)
    
    # Инициализация
    if 'generator' not in st.session_state:
        st.session_state.generator = None
    
    # Настройки в сайдбаре
    with st.sidebar:
        st.header("⚙️ Настройки")
        
        duration = st.slider(
            "Длительность (секунды)",
            min_value=5,
            max_value=30,
            value=10,
            step=5
        )
        
        method = st.radio(
            "Метод генерации",
            ["Простой (октавы)", "FFT метод"]
        )
        
        if method == "FFT метод":
            alpha = st.slider(
                "Характер шума (α)",
                min_value=0.5,
                max_value=1.5,
                value=1.0,
                step=0.1,
                help="1.0 = розовый, 0.5 = ближе к белому, 1.5 = ближе к коричневому"
            )
        
        use_envelope = st.checkbox("Плавные переходы", value=True)
        
        if use_envelope:
            fade_time = st.slider(
                "Время перехода (сек)",
                min_value=0.5,
                max_value=2.0,
                value=1.0,
                step=0.5
            )
        
        volume = st.slider(
            "Громкость",
            min_value=0.1,
            max_value=1.0,
            value=0.5,
            step=0.1
        )
        
        sample_rate = st.selectbox(
            "Качество",
            [22050, 44100],
            index=0,
            format_func=lambda x: f"{x} Гц ({'Стандарт' if x == 22050 else 'Высокое'})"
        )
    
    # Основной интерфейс
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.button("🎲 Генерировать розовый шум", type="primary", use_container_width=True):
            with st.spinner("Создание розового шума..."):
                # Создаем генератор
                st.session_state.generator = PinkNoiseGenerator(duration, sample_rate)
                
                # Генерируем шум
                if method == "Простой (октавы)":
                    st.session_state.generator.generate_pink_noise_simple()
                else:
                    st.session_state.generator.generate_pink_noise_fft(alpha)
                
                # Применяем огибающую
                if use_envelope:
                    st.session_state.generator.apply_envelope(fade_time, fade_time)
                
                st.success("✅ Готово!")
    
    with col2:
        st.info(f"⏱️ {duration} сек | 🔊 {int(volume*100)}%")
    
    # Воспроизведение и визуализация
    if st.session_state.generator and st.session_state.generator.noise is not None:
        
        # Аудио плеер
        st.subheader("🎧 Прослушивание")
        audio_html = st.session_state.generator.get_audio_html(volume)
        if audio_html:
            st.markdown(audio_html, unsafe_allow_html=True)
        
        # Визуализация
        st.subheader("📊 Визуализация")
        
        tab1, tab2 = st.tabs(["Форма волны", "Спектр"])
        
        with tab1:
            # График формы волны
            samples_to_show = min(sample_rate * 2, len(st.session_state.generator.noise))
            time_axis = np.linspace(0, samples_to_show/sample_rate, samples_to_show)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=time_axis,
                y=st.session_state.generator.noise[:samples_to_show],
                mode='lines',
                name='Розовый шум',
                line=dict(color='pink', width=1)
            ))
            fig.update_layout(
                xaxis_title="Время (сек)",
                yaxis_title="Амплитуда",
                height=300,
                showlegend=False,
                margin=dict(l=0, r=0, t=0, b=0)
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            # Спектр
            noise_data = st.session_state.generator.noise
            fft_data = np.fft.rfft(noise_data)
            freqs = np.fft.rfftfreq(len(noise_data), 1/sample_rate)
            
            # Усредняем спектр для сглаживания
            window_size = len(freqs) // 100
            if window_size > 1:
                power = np.abs(fft_data) ** 2
                # Простое скользящее среднее
                power_smooth = np.convolve(power, np.ones(window_size)/window_size, mode='valid')
                freqs_smooth = freqs[:len(power_smooth)]
                
                # Ограничиваем частоты для визуализации
                mask = (freqs_smooth > 20) & (freqs_smooth < 2000)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=freqs_smooth[mask],
                    y=10 * np.log10(power_smooth[mask] + 1e-10),
                    mode='lines',
                    name='Спектр мощности',
                    line=dict(color='blue', width=2)
                ))
                
                # Добавляем теоретическую линию 1/f
                theoretical_1f = -10 * np.log10(freqs_smooth[mask] / freqs_smooth[mask][0])
                theoretical_1f = theoretical_1f - np.mean(theoretical_1f) + np.mean(10 * np.log10(power_smooth[mask] + 1e-10))
                
                fig.add_trace(go.Scatter(
                    x=freqs_smooth[mask],
                    y=theoretical_1f,
                    mode='lines',
                    name='Идеальный 1/f',
                    line=dict(color='red', width=1, dash='dash')
                ))
                
                fig.update_xaxes(type="log", title="Частота (Гц)")
                fig.update_yaxes(title="Мощность (дБ)")
                fig.update_layout(
                    height=300,
                    showlegend=True,
                    margin=dict(l=0, r=0, t=0, b=0)
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Скачивание
        st.subheader("💾 Сохранение")
        
        col1, col2 = st.columns(2)
        with col1:
            wav_buffer = st.session_state.generator.save_wav(volume)
            if wav_buffer:
                st.download_button(
                    label="📥 Скачать WAV",
                    data=wav_buffer,
                    file_name=f"pink_noise_{duration}s.wav",
                    mime="audio/wav",
                    use_container_width=True
                )
        
        with col2:
            file_size = duration * sample_rate * 2 / 1024 / 1024
            st.info(f"Размер: ~{file_size:.1f} МБ")
    
    # Информация
    with st.expander("ℹ️ О розовом шуме"):
        st.markdown("""
        **Розовый шум** имеет спектр мощности, обратно пропорциональный частоте (1/f).
        
        **Применение:**
        - 🧘 **Медитация** - создает спокойный фон
        - 😴 **Сон** - маскирует посторонние звуки
        - 🎯 **Фокусировка** - помогает концентрации
        - 👶 **Для младенцев** - успокаивающий эффект
        
        **Природные примеры:**
        - Шум дождя
        - Звук водопада
        - Шелест листьев
        - Морской прибой
        
        **Рекомендации:**
        - Начинайте с низкой громкости
        - Используйте 10-30 минут для медитации
        - Для сна можно оставить на всю ночь
        """)

if __name__ == "__main__":
    main()
