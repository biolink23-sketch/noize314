import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import io


class BakSneppen:
    """
    Модель Бака-Снеппена для генерации розового шума.
    
    Модель основана на самоорганизованной критичности (SOC),
    где эволюционирует система видов с "приспособленностью" (fitness).
    """
    
    def __init__(self, n_species=1000, threshold=0.5):
        self.n_species = n_species
        self.threshold = threshold
        self.fitness = np.random.random(n_species)
        self.avalanche_sizes = []
        
    def evolve_step(self):
        """Один шаг эволюции модели Бака-Снеппена"""
        # Находим вид с минимальной приспособленностью
        min_idx = np.argmin(self.fitness)
        
        # Мутируем этот вид и его соседей
        self.fitness[min_idx] = np.random.random()
        self.fitness[(min_idx - 1) % self.n_species] = np.random.random()
        self.fitness[(min_idx + 1) % self.n_species] = np.random.random()
        
        return min_idx
    
    def generate_avalanche_series(self, n_steps=10000):
        """Генерация серии лавин (avalanches) - основы розового шума"""
        avalanche_series = []
        
        for _ in range(n_steps):
            min_idx = self.evolve_step()
            # Записываем позицию мутации как вклад в сигнал
            avalanche_series.append(min_idx)
        
        return np.array(avalanche_series)


def generate_pink_noise_bak_sneppen(duration=5, sample_rate=44100):
    """
    Генерация розового шума через модель Бака-Снеппена
    
    Args:
        duration: длительность в секундах
        sample_rate: частота дискретизации
    
    Returns:
        numpy array с розовым шумом
    """
    n_samples = int(duration * sample_rate)
    
    # Создаем модель Бака-Снеппена
    bs = BakSneppen(n_species=500, threshold=0.5)
    
    # Генерируем базовую серию
    base_series = bs.generate_avalanche_series(n_steps=n_samples)
    
    # Нормализуем к диапазону [-1, 1]
    pink_noise = (base_series - np.mean(base_series)) / np.std(base_series)
    pink_noise = np.clip(pink_noise * 0.3, -1, 1)
    
    return pink_noise


def analyze_spectrum(signal_data, sample_rate):
    """Анализ спектра сигнала для проверки розового шума"""
    freqs, psd = signal.welch(signal_data, sample_rate, nperseg=1024)
    return freqs, psd


def main():
    st.set_page_config(page_title="Генератор розового шума (Бак-Снеппен)", layout="wide")
    
    st.title("🎵 Генератор розового шума")
    st.markdown("### Модель Бака-Снеппена (Bak-Sneppen)")
    
    st.markdown("""
    **Модель Бака-Снеппена** — это математическая модель самоорганизованной критичности (SOC), 
    изначально разработанная для описания эволюционных процессов. Модель демонстрирует 
    степенное распределение событий, что приводит к генерации розового шума (1/f шума).
    """)
    
    # Боковая панель с параметрами
    st.sidebar.header("Параметры генерации")
    
    duration = st.sidebar.slider("Длительность (секунды)", 1, 10, 5)
    sample_rate = st.sidebar.selectbox("Частота дискретизации", 
                                       [22050, 44100, 48000], index=1)
    n_species = st.sidebar.slider("Количество видов в модели", 100, 2000, 500, step=100)
    
    # Кнопка генерации
    if st.sidebar.button("🎲 Генерировать розовый шум", type="primary"):
        with st.spinner("Генерация розового шума..."):
            # Генерация
            pink_noise = generate_pink_noise_bak_sneppen(duration, sample_rate)
            
            # Визуализация
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Временная форма сигнала")
                fig1, ax1 = plt.subplots(figsize=(10, 4))
                time = np.linspace(0, duration, len(pink_noise))
                ax1.plot(time, pink_noise, color='#FF69B4', linewidth=0.5)
                ax1.set_xlabel("Время (с)")
                ax1.set_ylabel("Амплитуда")
                ax1.set_title("Розовый шум (временная область)")
                ax1.grid(True, alpha=0.3)
                st.pyplot(fig1)
                plt.close()
            
            with col2:
                st.subheader("Спектральная плотность мощности")
                freqs, psd = analyze_spectrum(pink_noise, sample_rate)
                
                fig2, ax2 = plt.subplots(figsize=(10, 4))
                ax2.loglog(freqs[1:], psd[1:], color='#FF1493', linewidth=2)
                ax2.set_xlabel("Частота (Гц)")
                ax2.set_ylabel("Мощность")
                ax2.set_title("Спектр (должен иметь наклон ~-1 для розового шума)")
                ax2.grid(True, alpha=0.3, which="both")
                st.pyplot(fig2)
                plt.close()
            
            # Статистика
            st.subheader("📊 Статистика сигнала")
            col3, col4, col5 = st.columns(3)
            
            with col3:
                st.metric("Среднее значение", f"{np.mean(pink_noise):.4f}")
            with col4:
                st.metric("СКО", f"{np.std(pink_noise):.4f}")
            with col5:
                st.metric("Количество сэмплов", f"{len(pink_noise):,}")
            
            # Сохранение в WAV
            st.subheader("💾 Скачать аудиофайл")
            
            # Конвертация в int16 для WAV
            pink_noise_int16 = np.int16(pink_noise * 32767)
            
            # Создание WAV в памяти
            wav_buffer = io.BytesIO()
            wavfile.write(wav_buffer, sample_rate, pink_noise_int16)
            wav_buffer.seek(0)
            
            st.download_button(
                label="⬇️ Скачать WAV файл",
                data=wav_buffer,
                file_name=f"pink_noise_bak_sneppen_{duration}s.wav",
                mime="audio/wav"
            )
            
            # Объяснение модели
            with st.expander("ℹ️ Как работает модель Бака-Снеппена?"):
                st.markdown("""
                1. **Инициализация**: Создается массив из N "видов", каждому присваивается случайное значение "приспособленности" (fitness) от 0 до 1.
                
                2. **Эволюция**: На каждом шаге:
                   - Находится вид с минимальной приспособленностью
                   - Этот вид и два его соседа "мутируют" — получают новые случайные значения fitness
                
                3. **Самоорганизованная критичность**: Система естественным образом эволюционирует к критическому состоянию, 
                   где мутации вызывают "лавины" изменений различного масштаба.
                
                4. **Розовый шум**: Распределение размеров этих лавин следует степенному закону, что приводит к спектру 1/f — 
                   характеристике розового шума.
                
                **Применение**: Розовый шум используется в аудиопроизводстве, тестировании акустических систем, 
                моделировании природных процессов и даже в исследованиях сна.
                """)
    
    else:
        st.info("👈 Настройте параметры на боковой панели и нажмите кнопку генерации")
        
        # Демонстрационная визуализация модели
        st.subheader("🧬 Визуализация модели Бака-Снеппена")
        
        bs_demo = BakSneppen(n_species=100)
        fitness_history = [bs_demo.fitness.copy()]
        
        for _ in range(50):
            bs_demo.evolve_step()
            fitness_history.append(bs_demo.fitness.copy())
        
        fig3, ax3 = plt.subplots(figsize=(12, 6))
        fitness_array = np.array(fitness_history)
        im = ax3.imshow(fitness_array.T, aspect='auto', cmap='plasma', interpolation='nearest')
        ax3.set_xlabel("Шаг эволюции")
        ax3.set_ylabel("ID вида")
        ax3.set_title("Эволюция приспособленности видов (яркие = высокая приспособленность)")
        plt.colorbar(im, ax=ax3, label="Fitness")
        st.pyplot(fig3)
        plt.close()


if __name__ == "__main__":
    main()
