<template>
  <div class="waifu">
    <div class="waifu-tips"></div>
    <canvas id="live2d" class="live2d"></canvas>
    <div class="waifu-tool">
      <span class="fui-home"></span>
      <span class="fui-chat"></span>
      <span class="fui-eye"></span>
      <span class="fui-user"></span>
      <span class="fui-photo"></span>
      <span class="fui-info-circle"></span>
      <span class="fui-cross"></span>
    </div>
  </div>
</template>

<script setup>
import { onMounted, onBeforeUnmount } from "vue";

// 用于存储加载的脚本，便于清理
const loadedScripts = [];

// 动态加载JS和CSS
function loadScript(src) {
  return new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.src = src;
    script.onload = resolve;
    script.onerror = reject;
    document.head.appendChild(script);
    loadedScripts.push(script);
  });
}

function loadCSS(href) {
  return new Promise((resolve) => {
    const link = document.createElement('link');
    link.rel = 'stylesheet';
    link.type = 'text/css';
    link.href = href;
    link.onload = resolve;
    document.head.appendChild(link);
    loadedScripts.push(link); // 使用同一数组存储以便清理
  });
}

onMounted(async () => {
  try {
    // 先加载CSS确保样式优先应用
    await loadCSS("/live2d/assets/waifu.css");
    
    // 按顺序加载所需脚本
    await loadScript("/live2d/assets/jquery.min.js");
    await loadScript(
      "https://cdnjs.cloudflare.com/ajax/libs/jquery-easing/1.4.1/jquery.easing.min.js"
    );
    await loadScript("/live2d/assets/jquery-ui.min.js");
    await loadScript("/live2d/assets/waifu-tips.js");
    await loadScript("/live2d/assets/live2d.js");

    // 初始化Live2D配置
    live2d_settings['modelId'] = 1;                  // 默认模型 ID
    live2d_settings['modelTexturesId'] = 0;         // 默认材质 ID
    live2d_settings['modelStorage'] = false;         // 不储存模型 ID
    live2d_settings['canCloseLive2d'] = true;       // 关闭看板娘 按钮
    live2d_settings['canTurnToHomePage'] = false;    // 隐藏 返回首页 按钮
    live2d_settings['waifuSize'] = '430x430';        // 看板娘大小
    live2d_settings['waifuTipsSize'] = '500x130';    // 提示框大小
    live2d_settings['waifuFontSize'] = '30px';       // 提示框字体大小
    live2d_settings['waifuToolFont'] = '40px';       // 工具栏字体
    live2d_settings['waifuToolLine'] = '60px';       // 工具栏行高
    live2d_settings['waifuToolTop'] = '-90px';       // 工具栏顶部边距
    live2d_settings['waifuDraggable'] = 'disable';    // 拖拽样式
    window.live2d_settings['waifuEdgeSide'] = 'left:-90';
    // 初始化模型
    if (typeof window.initModel === 'function') {
      window.initModel("/live2d/assets/waifu-tips.json");
    }
  } catch (error) {
    console.error("Failed to load Live2D:", error);
  }
});

onBeforeUnmount(() => {
  // 清理加载的脚本
  loadedScripts.forEach((element) => {
    element.parentNode?.removeChild(element);
  });
});
</script>

<style>
</style>