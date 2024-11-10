<script lang="ts" setup>
import { useElementSize, useWindowScroll, useWindowSize } from '@vueuse/core'
import { computed, onMounted, ref, shallowRef, watch } from 'vue'
import { useData } from 'vuepress-theme-plume/composables'

const body = shallowRef<HTMLElement | null>()
const { height: bodyHeight } = useElementSize(body)
const { height: windowHeight } = useWindowSize()
onMounted(() => {
  body.value = document.body
})

const { page } = useData()
const { y } = useWindowScroll()
const show = computed(() => {
  if (bodyHeight.value < windowHeight.value) return false
  return y.value > windowHeight.value / 2
})

function handleClick() {
  window.scrollTo({ top: 0, behavior: 'smooth' })
}

const progress = computed(() => {
  return (y.value / (bodyHeight.value - windowHeight.value)) * 100
})

const stroke = computed(() => {
  return `calc(${Math.PI * progress.value}% - ${4 * Math.PI}px) calc(${Math.PI * 100}% - ${4 * Math.PI}px)`
})
</script>

<template>
  <Transition name="fade">
    <button
      v-show="show"
      type="button"
      class="vp-back-to-top"
      aria-label="back to top"
      @click="handleClick"
    >
      <span class="icon vpi-back-to-top" />
      <svg aria-hidden="true">
        <circle cx="50%" cy="50%" fill="none" :style="{ 'stroke-dasharray': stroke }" />
      </svg>
    </button>
  </Transition>
</template>

<style scoped>
.vp-back-to-top {
  position: fixed;
  inset-inline-end: 1rem;
  right: 24px;
  bottom: calc(var(--vp-footer-height, 82px) - 18px);
  z-index: var(--vp-z-index-back-to-top);
  width: 36px;
  height: 36px;
  background-color: var(--vp-c-bg);
  border-radius: 100%;
  box-shadow: var(--vp-shadow-2);
  transition: background-color var(--vp-t-color), box-shadow var(--vp-t-color);
  overflow: hidden; /* 确保图标和圆圈不会溢出 */
}

.vp-back-to-top .icon {
  width: 18px; /* 图标的大小 */
  height: 18px;
  color: var(--vp-c-text-3);
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%); /* 居中显示图标 */
}

.vp-back-to-top svg {
  width: 100%;
  height: 100%;
}

.vp-back-to-top svg circle {
  stroke: var(--vp-c-brand-2);
  stroke-width: 4px;
  transform: rotate(-90deg);
  transform-origin: 50% 50%;
  r: 16; /* 圆圈半径 */
}

@media (min-width: 768px) {
  .vp-back-to-top {
    bottom: calc(var(--vp-footer-height, 88px) - 24px);
    width: 48px;
    height: 48px;
  }

  .vp-back-to-top .icon {
    width: 24px; /* 图标的大小 */
    height: 24px;
  }

  .vp-back-to-top svg circle {
    r: 22; /* 圆圈半径 */
  }
}

.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.5s ease;
}

.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}

@media print {
  .vp-back-to-top {
    display: none;
  }
}
</style>
