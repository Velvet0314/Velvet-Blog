import './styles/index.css'
import RepoCard from 'vuepress-theme-plume/features/RepoCard.vue'
import Live2D from "./components/Live2D.vue";
import { onMounted, createApp } from "vue";
import { defineClientConfig } from "vuepress/client";

export default defineClientConfig({
  enhance({ app }) {
    app.component("RepoCard", RepoCard);
    app.component("Live2D", Live2D);
  },
  setup() {
    onMounted(() => {
      // 确保Live2D组件被全局挂载
      if (!document.querySelector('.live2d-container')) {
        const container = document.createElement('div');
        container.className = 'live2d-container';
        
        // 直接挂载到body而不是#app，避免被其他元素约束
        document.body.appendChild(container);
        
        // 使用Vue动态创建Live2D组件实例
        const live2dApp = createApp(Live2D);
        live2dApp.mount(container);
      }
    });
  },
});