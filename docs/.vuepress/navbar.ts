import { defineNavbarConfig } from 'vuepress-theme-plume'

export const navbar = defineNavbarConfig([
  { text: "首页", link: "/", icon: "material-symbols:cottage-rounded" },
  {
    text: "项目",
    link: "/projlist/",
    icon: "material-symbols:event-available-rounded",
  },
  { text: "博客", link: "/blog/", icon: "material-symbols:breaking-news" },
  {
    text: "灵感",
    icon: "material-symbols:lightbulb-2-rounded",
    items: [
      {
        text: "Paper",
        icon: "fa-solid:paper-plane",
        items: [
          {
            text: "arXiv.org e-Print archive",
            link: "https://arxiv.org/",
            icon: "simple-icons:arxiv",
          },
        ],
      },
    ],
  },
  {
    text: "笔记",
    icon: "material-symbols:book-4-spark-rounded",
    items: [
      { 
        text: "机器学习",
        items:[
          {
            text: "Stanford CS229 讲义",
            link: "/ML/",
            activeMatch: '^/ML/',
            icon: "icon-park-solid:notebook"
          }
        ],
      },
      { 
        text: "计算机基础",
        items:[
          {
            text: "浙江大学第2版 数据结构",
            link: "/DS/",
            activeMatch: '^/DS/',
            icon: "bxs:tree"
          }
        ],
      },
    ],
  },
]);
